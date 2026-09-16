"""LLM 适配器模块。

提供 LLM 调用的抽象接口和 OpenAI 兼容实现。
"""

import asyncio
import logging
import os
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

from tenacity import retry, wait_random_exponential, stop_after_attempt, RetryError

logger = logging.getLogger(__name__)


def _thinking_disabled_body() -> dict:
    """禁用 thinking 的 extra_body：同时携带 OpenRouter 与 DashScope 两种风格。

    DashScope 系网关 enable_thinking 默认为 true，且对非流式调用强制要求
    enable_thinking=false，否则 400；OpenAI-compat 网关忽略不认识的
    extra_body 键，因此两种形式可以安全并存。
    """
    return {"thinking": {"type": "disabled"}, "enable_thinking": False}


class LLMRequestTimeoutError(TimeoutError):
    """单次 LLM 请求超过总时限（request_timeout）。

    httpx 的 read timeout 是"字节间隔"级别：网关僵死时定期发心跳空白包
    会不断重置该计时器，导致请求永久挂起。此异常表示墙钟总时限已到，
    由 tenacity 捕获后按普通失败重试。
    """


@dataclass
class LLMUsage:
    """LLM 调用用量信息。"""
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class LLMResponse:
    """LLM 调用返回结果，包含文本内容和用量。"""
    content: str
    usage: LLMUsage = None
    tool_calls: list[dict] = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = LLMUsage()
        if self.tool_calls is None:
            self.tool_calls = []


class LLMAdapter(ABC):
    """LLM 适配器抽象基类。"""

    @abstractmethod
    async def agen(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """异步调用 LLM，返回 LLMResponse（含 usage）。"""

    def call(self, messages: list[dict], **kwargs) -> LLMResponse:
        """同步调用 LLM（默认实现用 asyncio.run）。"""
        import asyncio
        return asyncio.run(self.agen(messages, **kwargs))


class OpenAICompatAdapter(LLMAdapter):
    """OpenAI 兼容 API 适配器。

    支持 GPT/DeepSeek/Ollama/vLLM 等所有 OpenAI 兼容 API。

    同时维护同步和异步两个客户端：
    - call()  使用同步 OpenAI 客户端，避免 asyncio.run() 临时事件循环
              关闭后 httpx/anyio 延迟回调导致的 RuntimeError
    - agen()  使用异步 AsyncOpenAI 客户端，用于 Graph.arun() 路径
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
        timeout: float = 120.0,
        request_timeout: float = 900.0,
        json_mode: bool = False,
        disable_thinking: bool = False,
        thinking_budget: int = 0,
        stream: bool = False,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.request_timeout = request_timeout
        self.json_mode = json_mode
        self.disable_thinking = disable_thinking
        self.thinking_budget = thinking_budget
        self.stream = stream
        self._client = None       # 异步客户端（agen 使用）
        self._sync_client = None  # 同步客户端（call 使用）
        self._model_context_length: Optional[int] = None  # 缓存模型上下文长度

    def _http_timeout(self) -> float:
        """httpx 客户端超时。流式模式下思考模型的请求可能在网关排队
        或长时间不吐 chunk（120s 默认太短，实测 >250s），放宽到与
        request_timeout 一致；总墙钟仍由 deadline 包装器兜底。"""
        if self.stream:
            return max(self.timeout, self.request_timeout)
        return self.timeout

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            import httpx
            self._client = AsyncOpenAI(
                api_key=self.api_key or "sk-placeholder",
                base_url=self.base_url,
                timeout=httpx.Timeout(self._http_timeout(), connect=30.0),
            )
        return self._client

    def _get_sync_client(self):
        if self._sync_client is None:
            from openai import OpenAI
            import httpx
            self._sync_client = OpenAI(
                api_key=self.api_key or "sk-placeholder",
                base_url=self.base_url,
                timeout=httpx.Timeout(self._http_timeout(), connect=30.0),
            )
        return self._sync_client

    def _extract_response_content(self, response) -> tuple:
        """从 API 响应中提取内容和诊断信息。

        Returns:
            (content: str, finish_reason: str, reasoning_content: str | None)
        """
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", "unknown") or "unknown"
        # DeepSeek 等推理模型的 thinking/reasoning 内容在独立字段
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        return content, finish_reason, reasoning_content

    def _log_empty_content(self, response, elapsed: float, finish_reason: str,
                           reasoning_content, max_tokens_effective: int) -> None:
        """记录空内容响应的详细诊断日志。"""
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        max_tok = max_tokens_effective
        has_reasoning = bool(reasoning_content)
        reasoning_len = len(reasoning_content) if reasoning_content else 0

        if finish_reason == "length" and completion_tokens >= max_tok:
            logger.warning(
                "LLM call returned empty content (TOKEN EXHAUSTION): "
                "model=%s elapsed=%.1fs prompt=%dP/%dC max_tokens=%d "
                "finish_reason=%s has_reasoning_content=%s reasoning_len=%d. "
                "All %d completion tokens consumed but no output produced — "
                "reasoning/thinking phase likely exhausted the token budget. "
                "Consider increasing max_tokens or reducing prompt size.",
                self.model, elapsed, prompt_tokens, completion_tokens,
                max_tok, finish_reason, has_reasoning, reasoning_len,
                completion_tokens,
            )
        else:
            logger.warning(
                "LLM call returned empty content: model=%s elapsed=%.1fs "
                "prompt=%dP/%dC max_tokens=%d finish_reason=%s "
                "has_reasoning_content=%s reasoning_len=%d",
                self.model, elapsed, prompt_tokens, completion_tokens,
                max_tok, finish_reason, has_reasoning, reasoning_len,
            )

    @staticmethod
    def _parse_max_model_len_from_error(error: Exception) -> Optional[int]:
        """从 vLLM/OpenAI 的 BadRequestError 中解析 max_model_len / max_total_tokens。"""
        text = str(error)
        for pattern in (r"max_model_len[=:]\s*(\d+)", r"max_total_tokens[=:]\s*(\d+)"):
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        return None

    def _clamp_max_tokens_for_bad_request(
        self, error: Exception, current_max_tokens: int
    ) -> Optional[int]:
        """如果错误是 max_tokens 超过模型上下文上限，返回调整后的 max_tokens；否则 None。"""
        err_text = str(error).lower()
        # DashScope 风格：直接给出 max_tokens 的合法上限（completion 上限，
        # 非总上下文长度），按该上限裁剪即可。注意流式模式下网关把错误塞进
        # 流里，SDK 抛出的是 openai.APIError 而非 BadRequestError，因此这个
        # 分支必须在异常类型检查之前、对任意异常生效。
        match = re.search(r"range of max_tokens.*?\[1,\s*(\d+)\]", err_text)
        if match:
            limit = int(match.group(1))
            if limit >= current_max_tokens:
                return None
            logger.warning(
                "LLM call: max_tokens=%d exceeds provider limit (%d), "
                "retrying with max_tokens=%d",
                current_max_tokens, limit, limit,
            )
            return limit
        from openai import BadRequestError
        if not isinstance(error, BadRequestError):
            return None
        if "max_tokens" not in err_text or "max_model_len" not in err_text:
            return None
        model_limit = self._parse_max_model_len_from_error(error)
        if model_limit is None:
            return None
        # 留出一半上下文给 prompt，避免再次触发上限
        new_max = min(current_max_tokens, max(model_limit // 2, 1024))
        if new_max >= current_max_tokens:
            return None
        logger.warning(
            "LLM call: max_tokens=%d exceeds model context length (%d), "
            "retrying with max_tokens=%d",
            current_max_tokens, model_limit, new_max,
        )
        return new_max

    def _fetch_model_max_context_length(self) -> Optional[int]:
        """尝试从 /v1/models 获取模型上下文长度（vLLM 返回 max_model_len）。

        结果会被缓存，避免每次调用都查询。
        """
        if self._model_context_length is not None:
            return self._model_context_length
        if not self.base_url:
            return None
        try:
            import httpx
            url = self.base_url.rstrip("/") + "/models"
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            with httpx.Client(timeout=10.0) as http:
                resp = http.get(url, headers=headers)
            if resp.status_code != 200:
                return None
            data = resp.json()
            for m in data.get("data", []):
                if m.get("id") == self.model:
                    limit = m.get("max_model_len")
                    if isinstance(limit, int):
                        self._model_context_length = limit
                        return limit
                    break
            return None
        except Exception:
            return None

    def _clamp_max_tokens_to_model_limit(self, max_tokens: int) -> int:
        """根据模型上下文长度预裁剪 max_tokens，避免触发 BadRequestError。"""
        limit = self._fetch_model_max_context_length()
        if limit is None:
            return max_tokens
        # 留一半上下文给 prompt，降低越界概率
        safe = max(limit // 2, 1024)
        if max_tokens > safe:
            logger.info(
                "LLM call: clamping max_tokens %d -> %d based on model context length %d",
                max_tokens, safe, limit,
            )
            return safe
        return max_tokens

    def _do_create(self, client, create_kwargs: dict):
        """同步发起请求；stream=True 时在 deadline 线程内消费完整流。"""
        resp = client.chat.completions.create(**create_kwargs)
        if create_kwargs.get("stream"):
            return self._collect_stream(resp)
        return resp

    async def _ado_create(self, client, create_kwargs: dict):
        """异步发起请求；stream=True 时消费完整异步流。"""
        resp = await client.chat.completions.create(**create_kwargs)
        if create_kwargs.get("stream"):
            return await self._acollect_stream(resp)
        return resp

    @staticmethod
    def _collect_stream(resp):
        """消费同步流式响应，拼装成与非流式一致的响应形状。

        content/reasoning_content 逐 chunk 拼接；tool_calls 按 index
        重组分片（id/name 取首个非空，arguments 拼接）；usage 取自带
        （需 stream_options={"include_usage": True}，通常在收尾空 chunk）。
        """
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        finish_reason = None
        usage = None
        for chunk in resp:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = tool_acc.setdefault(
                    tc.index, {"id": None, "name": "", "args": []}
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"].append(fn.arguments)
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
        tool_calls = [
            SimpleNamespace(
                id=slot["id"],
                function=SimpleNamespace(
                    name=slot["name"], arguments="".join(slot["args"])
                ),
            )
            for _, slot in sorted(tool_acc.items())
        ] or None
        message = SimpleNamespace(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=message, finish_reason=finish_reason or "unknown"
            )],
            usage=usage,
        )

    @staticmethod
    async def _acollect_stream(resp):
        """_collect_stream 的异步版本。"""
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        finish_reason = None
        usage = None
        async for chunk in resp:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = tool_acc.setdefault(
                    tc.index, {"id": None, "name": "", "args": []}
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"].append(fn.arguments)
            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason
        tool_calls = [
            SimpleNamespace(
                id=slot["id"],
                function=SimpleNamespace(
                    name=slot["name"], arguments="".join(slot["args"])
                ),
            )
            for _, slot in sorted(tool_acc.items())
        ] or None
        message = SimpleNamespace(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts) or None,
            tool_calls=tool_calls,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=message, finish_reason=finish_reason or "unknown"
            )],
            usage=usage,
        )

    def _create_with_deadline(self, client, create_kwargs: dict):
        """以墙钟总时限调用 chat.completions.create。

        httpx 的 read timeout 会在收到任意字节（包括网关僵死时的心跳空白包）
        后重置，无法覆盖"请求永久不返回"的场景。这里把请求放进 daemon 线程，
        主线程用 queue.get(timeout=...) 等待：超时即抛 LLMRequestTimeoutError
        交给 tenacity 重试。挂起的 daemon 线程随进程退出被回收，不会拖死进程。
        request_timeout <= 0 时不启用，直接调用。
        """
        if not self.request_timeout or self.request_timeout <= 0:
            return self._do_create(client, create_kwargs)

        result_queue: queue.Queue = queue.Queue(maxsize=1)

        def _target():
            try:
                result_queue.put(("ok", self._do_create(client, create_kwargs)))
            except Exception as e:  # noqa: BLE001 - 原样转交主线程抛出
                result_queue.put(("err", e))

        threading.Thread(target=_target, daemon=True).start()
        try:
            status, value = result_queue.get(timeout=self.request_timeout)
        except queue.Empty:
            logger.warning(
                "LLM call: request deadline exceeded (%.0fs), model=%s",
                self.request_timeout, self.model,
            )
            raise LLMRequestTimeoutError(
                f"LLM request exceeded total deadline of {self.request_timeout:.0f}s "
                f"(model={self.model})"
            )
        if status == "err":
            raise value
        return value

    def _create_sync_with_deadline(self, create_kwargs: dict):
        """以同步客户端 + 阻塞式截止时间发起请求，供 run_in_executor 在子线程调用。

        截止时间通过 ``queue.get(timeout=...)`` 在子线程独立触发，不依赖
        事件循环的调度；这样即使事件循环被同步调用（run_python 子进程、
        SentenceTransformer 嵌入等）占住，僵尸请求也能按时超时，避免进程
        永久挂起。
        """
        return self._create_with_deadline(self._get_sync_client(), create_kwargs)

    async def _create_async_with_deadline(self, create_kwargs: dict):
        """在异步调用路径上提供不可取消阻塞的墙钟截止时间。

        真实 OpenAI SDK 的异步请求可能在取消时阻塞事件循环；更关键的是，
        依赖 ``await asyncio.sleep`` 的轮询在事件循环被同步调用占住时不会
        得到调度，导致僵尸请求的截止时间永不触发。因此对真实 SDK 把
        ``_create_with_deadline``（同步客户端 + 阻塞 ``queue.get(timeout)``）
        整体放入线程执行器：截止时间在子线程独立触发，不受事件循环调度
        影响。测试替身等自定义异步客户端则保留原协程轮询路径。
        """
        client = self._get_client()
        if not self.request_timeout or self.request_timeout <= 0:
            return await self._ado_create(client, create_kwargs)

        use_sync_sdk = client.__class__.__module__.split(".", 1)[0] == "openai"
        if use_sync_sdk:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None, self._create_sync_with_deadline, create_kwargs
            )

        result_queue: queue.Queue = queue.Queue(maxsize=1)

        def _target():
            try:
                response = asyncio.run(self._ado_create(client, create_kwargs))
                result_queue.put(("ok", response))
            except Exception as e:  # noqa: BLE001 - 原样转交事件循环
                result_queue.put(("err", e))

        threading.Thread(target=_target, daemon=True).start()
        deadline = time.monotonic() + self.request_timeout
        while True:
            try:
                status, value = result_queue.get_nowait()
                break
            except queue.Empty:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "LLM call: async request deadline exceeded (%.0fs), model=%s",
                        self.request_timeout, self.model,
                    )
                    raise LLMRequestTimeoutError(
                        f"Async LLM request exceeded total deadline of "
                        f"{self.request_timeout:.0f}s (model={self.model})"
                    )
                await asyncio.sleep(min(remaining, 0.05))
        if status == "err":
            raise value
        return value

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    def _call_with_retry(self, messages: list[dict], **kwargs) -> LLMResponse:
        """同步调用 LLM 的实际实现（带 tenacity 重试）。"""
        t_start = time.time()
        client = self._get_sync_client()
        max_tokens_effective = kwargs.get("max_tokens", self.max_tokens)
        max_tokens_effective = self._clamp_max_tokens_to_model_limit(max_tokens_effective)
        tools = kwargs.get("tools", None)
        create_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens_effective,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.stream:
            create_kwargs["stream"] = True
            # 流式的 usage 仅在显式请求后由收尾 chunk 携带
            create_kwargs["stream_options"] = {"include_usage": True}
        # tool calling 模式下不强制 json_object（二者互斥）
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"
        elif kwargs.get("json_mode", self.json_mode):
            create_kwargs["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            create_kwargs["extra_body"] = _thinking_disabled_body()
        elif self.thinking_budget > 0:
            create_kwargs["extra_body"] = {
                "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}
            }

        max_tokens_retried = False
        json_mode_retried = False
        for empty_retry in range(3):  # bad-request 降配重试 + token 耗尽 disable-thinking 回退
            try:
                response = self._create_with_deadline(client, create_kwargs)
            except Exception as e:
                if not max_tokens_retried:
                    new_max = self._clamp_max_tokens_for_bad_request(
                        e, max_tokens_effective)
                    if new_max is not None:
                        max_tokens_effective = new_max
                        create_kwargs["max_tokens"] = new_max
                        max_tokens_retried = True
                        t_start = time.time()
                        continue
                logger.warning("LLM call failed: %s", e)
                raise
            elapsed = time.time() - t_start

            content, finish_reason, reasoning_content = \
                self._extract_response_content(response)

            # 解析 tool_calls
            tool_calls = []
            choice = response.choices[0]
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })
                logger.info(
                    "LLM call: model=%s tool_calls=%s",
                    self.model, [tc["name"] for tc in tool_calls],
                )

            if content.strip() or tool_calls:
                break  # 正常响应（有内容或工具调用）

            # 空内容：记录诊断信息
            self._log_empty_content(
                response, elapsed, finish_reason, reasoning_content,
                max_tokens_effective,
            )

            # json_object 回退：部分网关（DashScope 百炼）在流式+思考模式下
            # 与 response_format=json_object 互斥，返回 finish=stop 的空正文；
            # 去掉 response_format 重试一次（agent 端本就有 JSON 解析兜底）。
            if create_kwargs.get("response_format") and not json_mode_retried:
                logger.warning(
                    "LLM call: empty content with response_format=%s, "
                    "retrying without it",
                    create_kwargs["response_format"],
                )
                create_kwargs.pop("response_format")
                json_mode_retried = True
                t_start = time.time()
                continue

            # Token 耗尽回退：max_tokens 不变，临时禁用 thinking 重试一次。
            # 推理模型可能在 thinking 阶段烧完全部 completion 配额导致正文为空；
            # 禁用 thinking 后正文通常远小于 max_tokens，无需上调配额。
            thinking_off = (
                create_kwargs.get("extra_body", {}).get("thinking", {}).get("type")
                == "disabled"
            )
            if finish_reason == "length" and not thinking_off:
                logger.warning(
                    "LLM call: retrying with thinking disabled "
                    "(max_tokens unchanged=%d)",
                    max_tokens_effective,
                )
                create_kwargs["extra_body"] = _thinking_disabled_body()
                t_start = time.time()
                continue

            break  # 无法恢复，返回空内容

        usage = LLMUsage()
        if response.usage:
            usage.prompt_tokens = response.usage.prompt_tokens or 0
            usage.completion_tokens = response.usage.completion_tokens or 0

        logger.info(
            "LLM call: model=%s elapsed=%.1fs prompt=%dP/%dC",
            self.model, elapsed, usage.prompt_tokens, usage.completion_tokens,
        )

        return LLMResponse(content=content, usage=usage, tool_calls=tool_calls)

    def call(self, messages: list[dict], **kwargs) -> LLMResponse:
        """同步调用 LLM，使用同步 OpenAI 客户端。

        在 ThreadPoolExecutor 的多线程场景下，每个线程直接调用同步 HTTP
        客户端，不创建临时 asyncio 事件循环，彻底避免 httpx/anyio 延迟
        回调在 loop 关闭后执行的 RuntimeError('Event loop is closed')。

        支持 OpenAI 原生 tool calling：传入 tools 参数时，自动设置
        tool_choice="auto"，响应中的 tool_calls 会解析到 LLMResponse.tool_calls。

        内置空内容回退：当 finish_reason 为 "length"（token 耗尽）时，
        max_tokens 不变，临时禁用 thinking 重试一次，避免推理模型在
        thinking 阶段耗尽 token 配额导致输出为空。

        当 tenacity 重试耗尽后，会把最后一次底层异常解包抛出，避免调用方
        收到难以阅读的 ``RetryError[<Future ...>]``。
        """
        try:
            return self._call_with_retry(messages, **kwargs)
        except RetryError as e:
            last_attempt = getattr(e, "last_attempt", None)
            if last_attempt is not None and getattr(last_attempt, "failed", False):
                last_exc = last_attempt.exception()
                if last_exc is not None:
                    raise last_exc from e
            raise

    async def aclose(self):
        """关闭异步客户端，释放 httpx 连接池资源。

        必须在事件循环关闭前调用，否则连接池的延迟清理回调会在 loop
        关闭后触发 RuntimeError('Event loop is closed')。

        AsyncOpenAI.close() 是同步方法，无法正确清理底层
        httpx.AsyncClient 的连接池；必须直接调用 aclose()。
        """
        if self._client is not None:
            raw_client = self._client
            self._client = None
            try:
                await raw_client._client.aclose()
            except RuntimeError:
                pass
            # 让 anyio transport 关闭时通过 call_soon 注册的
            # _call_connection_lost 回调有机会在 loop 关闭前执行
            try:
                await asyncio.sleep(0)
            except RuntimeError:
                pass
        self.close_sync()

    def close_sync(self):
        """关闭同步客户端，释放连接池资源。"""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None

    @retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
    async def _agen_with_retry(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """异步调用 LLM 的实际实现（带 tenacity 重试）。"""
        t_start = time.time()
        client = self._get_client()
        max_tokens_effective = max_tokens or self.max_tokens
        max_tokens_effective = self._clamp_max_tokens_to_model_limit(max_tokens_effective)
        tools = kwargs.get("tools", None)
        create_kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens_effective,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if self.stream:
            create_kwargs["stream"] = True
            # 流式的 usage 仅在显式请求后由收尾 chunk 携带
            create_kwargs["stream_options"] = {"include_usage": True}
        # tool calling 模式下不强制 json_object（二者互斥）
        if tools:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = "auto"
        elif kwargs.get("json_mode", self.json_mode):
            create_kwargs["response_format"] = {"type": "json_object"}
        if self.disable_thinking:
            create_kwargs["extra_body"] = _thinking_disabled_body()
        elif self.thinking_budget > 0:
            create_kwargs["extra_body"] = {
                "thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}
            }

        max_tokens_retried = False
        json_mode_retried = False
        for empty_retry in range(3):  # bad-request 降配重试 + token 耗尽 disable-thinking 回退
            try:
                response = await self._create_async_with_deadline(create_kwargs)
            except Exception as e:
                if not max_tokens_retried:
                    new_max = self._clamp_max_tokens_for_bad_request(
                        e, max_tokens_effective)
                    if new_max is not None:
                        max_tokens_effective = new_max
                        create_kwargs["max_tokens"] = new_max
                        max_tokens_retried = True
                        t_start = time.time()
                        continue
                logger.warning("LLM call failed: %s", e)
                raise
            elapsed = time.time() - t_start

            content, finish_reason, reasoning_content = \
                self._extract_response_content(response)

            # 解析 tool_calls
            tool_calls = []
            choice = response.choices[0]
            if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    })
                logger.info(
                    "LLM call: model=%s tool_calls=%s",
                    self.model, [tc["name"] for tc in tool_calls],
                )

            if content.strip() or tool_calls:
                break  # 正常响应（有内容或工具调用）

            # 空内容：记录诊断信息
            self._log_empty_content(
                response, elapsed, finish_reason, reasoning_content,
                max_tokens_effective,
            )

            # json_object 回退：部分网关（DashScope 百炼）在流式+思考模式下
            # 与 response_format=json_object 互斥，返回 finish=stop 的空正文；
            # 去掉 response_format 重试一次（agent 端本就有 JSON 解析兜底）。
            if create_kwargs.get("response_format") and not json_mode_retried:
                logger.warning(
                    "LLM call: empty content with response_format=%s, "
                    "retrying without it",
                    create_kwargs["response_format"],
                )
                create_kwargs.pop("response_format")
                json_mode_retried = True
                t_start = time.time()
                continue

            # Token 耗尽回退：max_tokens 不变，临时禁用 thinking 重试一次。
            # 推理模型可能在 thinking 阶段烧完全部 completion 配额导致正文为空；
            # 禁用 thinking 后正文通常远小于 max_tokens，无需上调配额。
            thinking_off = (
                create_kwargs.get("extra_body", {}).get("thinking", {}).get("type")
                == "disabled"
            )
            if finish_reason == "length" and not thinking_off:
                logger.warning(
                    "LLM call: retrying with thinking disabled "
                    "(max_tokens unchanged=%d)",
                    max_tokens_effective,
                )
                create_kwargs["extra_body"] = _thinking_disabled_body()
                t_start = time.time()
                continue

            break  # 无法恢复，返回空内容

        usage = LLMUsage()
        if response.usage:
            usage.prompt_tokens = response.usage.prompt_tokens or 0
            usage.completion_tokens = response.usage.completion_tokens or 0

        logger.info(
            "LLM call: model=%s elapsed=%.1fs prompt=%dP/%dC",
            self.model, elapsed, usage.prompt_tokens, usage.completion_tokens,
        )

        return LLMResponse(content=content, usage=usage, tool_calls=tool_calls)

    async def agen(
        self,
        messages: list[dict],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """异步调用 LLM，用于 Graph.arun() 路径。

        支持 OpenAI 原生 tool calling：传入 tools 参数时，自动设置
        tool_choice="auto"，响应中的 tool_calls 会解析到 LLMResponse.tool_calls。

        内置空内容回退：当 finish_reason 为 "length"（token 耗尽）时，
        max_tokens 不变，临时禁用 thinking 重试一次。

        当 tenacity 重试耗尽后，会把最后一次底层异常解包抛出，避免调用方
        收到难以阅读的 ``RetryError[<Future ...>]``。
        """
        try:
            return await self._agen_with_retry(
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            )
        except RetryError as e:
            last_attempt = getattr(e, "last_attempt", None)
            if last_attempt is not None and getattr(last_attempt, "failed", False):
                last_exc = last_attempt.exception()
                if last_exc is not None:
                    raise last_exc from e
            raise
