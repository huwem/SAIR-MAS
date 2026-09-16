"""LLM 适配器单元测试。"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from tenacity import Retrying, stop_after_attempt, wait_fixed

from mas_topo import _OpenAICompatRegistered
from mas_topo.llm import LLMRequestTimeoutError, OpenAICompatAdapter


def _make_response(content, finish_reason, reasoning=None,
                   prompt_tokens=10, completion_tokens=100):
    """构造 chat.completions 响应的简易替身。"""
    msg = SimpleNamespace(
        content=content, reasoning_content=reasoning, tool_calls=None,
    )
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class _MockSyncClient:
    """按队列返回响应并记录每次 create 调用的 kwargs。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _MockAsyncClient:
    """_MockSyncClient 的异步版本。"""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class TestOpenAICompatAdapter:
    """OpenAICompatAdapter 行为测试。"""

    def test_registered_adapter_preserves_request_timeout(self):
        """注册表包装器不能吞掉框架传入的总请求超时。"""
        adapter = _OpenAICompatRegistered(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            request_timeout=17.0,
        )
        assert adapter.request_timeout == 17.0

    def test_retry_error_is_unwrapped_to_root_cause(self):
        """当 tenacity 重试耗尽后，call() 应抛出底层异常而非 RetryError。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
        )

        class CustomAPIConnectionError(Exception):
            pass

        def always_fails(*args, **kwargs):
            raise CustomAPIConnectionError("connection refused")

        # 用真实 tenacity 触发 RetryError，然后验证 call() 解包后的异常
        try:
            for attempt in Retrying(wait=wait_fixed(0.01), stop=stop_after_attempt(3)):
                with attempt:
                    always_fails()
        except Exception as retry_error:
            # 将 RetryError 注入 adapter._call_with_retry 的调用路径
            original = adapter._get_sync_client
            mock_client = type(
                "MockClient",
                (),
                {
                    "chat": type(
                        "Chat",
                        (),
                        {
                            "completions": type(
                                "Completions",
                                (),
                                {"create": always_fails},
                            )()
                        },
                    )()
                },
            )()
            adapter._get_sync_client = lambda: mock_client
            try:
                with pytest.raises(CustomAPIConnectionError) as exc_info:
                    adapter.call([{"role": "user", "content": "hi"}])
                assert "connection refused" in str(exc_info.value)
            finally:
                adapter._get_sync_client = original

    @pytest.mark.asyncio
    async def test_async_retry_error_is_unwrapped_to_root_cause(self):
        """当 tenacity 重试耗尽后，agen() 应抛出底层异常而非 RetryError。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
        )

        class CustomAPIConnectionError(Exception):
            pass

        async def always_fails(*args, **kwargs):
            raise CustomAPIConnectionError("async connection refused")

        original = adapter._get_client
        mock_client = type(
            "MockAsyncClient",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {
                        "completions": type(
                            "Completions",
                            (),
                            {"create": always_fails},
                        )()
                    },
                )()
            },
        )()
        adapter._get_client = lambda: mock_client
        try:
            with pytest.raises(CustomAPIConnectionError) as exc_info:
                await adapter.agen([{"role": "user", "content": "hi"}])
            assert "async connection refused" in str(exc_info.value)
        finally:
            adapter._get_client = original

    def test_token_exhaustion_retries_with_thinking_disabled(self):
        """TOKEN EXHAUSTION 时：max_tokens 不变，临时禁用 thinking 重试一次。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            max_tokens=32000,
        )
        exhausted = _make_response(
            "", "length", reasoning="x" * 5000, completion_tokens=32000,
        )
        ok = _make_response('{"answer": 1}', "stop")
        client = _MockSyncClient([exhausted, ok])
        adapter._get_sync_client = lambda: client

        resp = adapter.call([{"role": "user", "content": "hi"}])

        assert resp.content == '{"answer": 1}'
        assert len(client.calls) == 2
        # 首次调用不带 extra_body
        assert "extra_body" not in client.calls[0]
        # 回退调用：禁用 thinking，且 max_tokens 保持不变
        assert client.calls[1]["extra_body"] == {"thinking": {"type": "disabled"}, "enable_thinking": False}
        assert client.calls[1]["max_tokens"] == 32000

    def test_token_exhaustion_no_retry_when_thinking_already_disabled(self):
        """thinking 已禁用时 token 耗尽不再重试，直接返回空内容。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            max_tokens=32000,
            disable_thinking=True,
        )
        exhausted = _make_response(
            "", "length", reasoning="x" * 5000, completion_tokens=32000,
        )
        client = _MockSyncClient([exhausted, exhausted])
        adapter._get_sync_client = lambda: client

        resp = adapter.call([{"role": "user", "content": "hi"}])

        assert resp.content == ""
        assert len(client.calls) == 1

    def test_empty_content_retries_without_response_format(self):
        """json_object 与流式思考在部分网关互斥返回空正文：去掉 response_format 重试一次。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
        )
        empty = _make_response("", "stop", completion_tokens=46)
        ok = _make_response('{"answer": 1}', "stop")
        client = _MockSyncClient([empty, ok])
        adapter._get_sync_client = lambda: client

        resp = adapter.call([{"role": "user", "content": "hi"}], json_mode=True)

        assert resp.content == '{"answer": 1}'
        assert len(client.calls) == 2
        assert client.calls[0]["response_format"] == {"type": "json_object"}
        assert "response_format" not in client.calls[1]

    @pytest.mark.asyncio
    async def test_agen_empty_content_retries_without_response_format(self):
        """agen() 的 json_object 空正文回退与 call() 行为一致。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
        )
        empty = _make_response("", "stop", completion_tokens=46)
        ok = _make_response('{"answer": 2}', "stop")
        client = _MockAsyncClient([empty, ok])
        adapter._get_client = lambda: client

        resp = await adapter.agen([{"role": "user", "content": "hi"}], json_mode=True)

        assert resp.content == '{"answer": 2}'
        assert len(client.calls) == 2
        assert client.calls[0]["response_format"] == {"type": "json_object"}
        assert "response_format" not in client.calls[1]

    def test_stream_mode_relaxes_http_timeout(self):
        """流式模式的 httpx 超时放宽到 request_timeout（思考模型长流 120s 不够）。"""
        streaming = OpenAICompatAdapter(
            model="test-model", api_key="sk-test",
            base_url="http://localhost:9999", stream=True,
        )
        assert streaming._http_timeout() == 900.0
        plain = OpenAICompatAdapter(
            model="test-model", api_key="sk-test",
            base_url="http://localhost:9999",
        )
        assert plain._http_timeout() == 120.0

    def test_disable_thinking_sends_enable_thinking_false(self):
        """disable_thinking=True 时同时发送 OpenRouter 与 DashScope 两种禁用形式。

        DashScope 系网关 enable_thinking 默认为 true，且对非流式调用强制
        要求 enable_thinking=false，否则 400（invalid_request_error）。
        """
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            disable_thinking=True,
        )
        ok = _make_response('{"answer": 1}', "stop")
        client = _MockSyncClient([ok])
        adapter._get_sync_client = lambda: client

        adapter.call([{"role": "user", "content": "hi"}])

        body = client.calls[0]["extra_body"]
        assert body["enable_thinking"] is False
        assert body["thinking"] == {"type": "disabled"}

    @pytest.mark.asyncio
    async def test_agen_disable_thinking_sends_enable_thinking_false(self):
        """异步路径与同步路径一致。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            disable_thinking=True,
        )
        ok = _make_response('{"answer": 1}', "stop")
        client = _MockAsyncClient([ok])
        adapter._get_client = lambda: client

        await adapter.agen([{"role": "user", "content": "hi"}])

        assert client.calls[0]["extra_body"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_agen_token_exhaustion_retries_with_thinking_disabled(self):
        """agen() 的 TOKEN EXHAUSTION 回退与 call() 行为一致。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            max_tokens=32000,
        )
        exhausted = _make_response(
            "", "length", reasoning="x" * 5000, completion_tokens=32000,
        )
        ok = _make_response('{"answer": 2}', "stop")
        client = _MockAsyncClient([exhausted, ok])
        adapter._get_client = lambda: client

        resp = await adapter.agen([{"role": "user", "content": "hi"}])

        assert resp.content == '{"answer": 2}'
        assert len(client.calls) == 2
        assert "extra_body" not in client.calls[0]
        assert client.calls[1]["extra_body"] == {"thinking": {"type": "disabled"}, "enable_thinking": False}
        assert client.calls[1]["max_tokens"] == 32000

    @pytest.mark.asyncio
    async def test_agen_times_out_when_request_hangs(self):
        """异步 Graph 路径也必须受总请求时限保护。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            request_timeout=0.03,
        )

        class HangingAsyncClient:
            def __init__(self):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self.create)
                )

            async def create(self, **kwargs):
                await asyncio.sleep(30)

        adapter._get_client = lambda: HangingAsyncClient()
        retry_cfg = adapter._agen_with_retry.retry
        old_wait, old_stop = retry_cfg.wait, retry_cfg.stop
        retry_cfg.wait = wait_fixed(0.01)
        retry_cfg.stop = stop_after_attempt(2)
        try:
            t0 = time.time()
            with pytest.raises(LLMRequestTimeoutError):
                await adapter.agen([{"role": "user", "content": "hi"}])
            assert time.time() - t0 < 1.0
        finally:
            retry_cfg.wait, retry_cfg.stop = old_wait, old_stop

    @pytest.mark.asyncio
    async def test_agen_sync_sdk_times_out_when_request_hangs(self):
        """真实 OpenAI SDK 路径（同步客户端 + 线程截止）也必须受总请求时限保护。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            request_timeout=0.03,
        )

        class FakeAsyncClient:
            pass
        FakeAsyncClient.__module__ = "openai"
        adapter._get_client = lambda: FakeAsyncClient()

        def hanging_create(self, **kwargs):
            time.sleep(30)
            return _make_response('{"answer": 1}', "stop")

        sync_client = type(
            "MockSyncClient",
            (),
            {
                "chat": type(
                    "Chat",
                    (),
                    {"completions": type("Completions", (), {"create": hanging_create})()},
                )()
            },
        )()
        adapter._get_sync_client = lambda: sync_client

        retry_cfg = adapter._agen_with_retry.retry
        old_wait, old_stop = retry_cfg.wait, retry_cfg.stop
        retry_cfg.wait = wait_fixed(0.01)
        retry_cfg.stop = stop_after_attempt(2)
        try:
            t0 = time.time()
            with pytest.raises(LLMRequestTimeoutError):
                await adapter.agen([{"role": "user", "content": "hi"}])
            assert time.time() - t0 < 1.0
        finally:
            retry_cfg.wait, retry_cfg.stop = old_wait, old_stop

    def test_call_times_out_when_request_hangs(self):
        """网关僵死（连接挂着但永不返回字节）时，call() 应在 request_timeout
        后抛 LLMRequestTimeoutError，而不是永久阻塞。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            request_timeout=0.3,
        )

        def hanging_create(**kwargs):
            time.sleep(30)
            return _make_response('{"answer": 1}', "stop")

        client = _MockSyncClient([None])
        client.chat.completions.create = hanging_create
        adapter._get_sync_client = lambda: client

        retry_cfg = adapter._call_with_retry.retry
        old_wait, old_stop = retry_cfg.wait, retry_cfg.stop
        retry_cfg.wait = wait_fixed(0.01)
        retry_cfg.stop = stop_after_attempt(2)
        try:
            t0 = time.time()
            with pytest.raises(LLMRequestTimeoutError):
                adapter.call([{"role": "user", "content": "hi"}])
            # 2 次尝试各 0.3s + 极短退避，绝不能等满 30s
            assert time.time() - t0 < 5.0
        finally:
            retry_cfg.wait, retry_cfg.stop = old_wait, old_stop

    def test_call_recovers_when_retry_is_fast(self):
        """第一次请求僵死超时后，tenacity 重试成功则正常返回内容。"""
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
            request_timeout=0.3,
        )
        calls = []

        def flaky_create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                time.sleep(30)
            return _make_response('{"answer": 7}', "stop")

        client = _MockSyncClient([None])
        client.chat.completions.create = flaky_create
        adapter._get_sync_client = lambda: client

        retry_cfg = adapter._call_with_retry.retry
        old_wait, old_stop = retry_cfg.wait, retry_cfg.stop
        retry_cfg.wait = wait_fixed(0.01)
        retry_cfg.stop = stop_after_attempt(3)
        try:
            resp = adapter.call([{"role": "user", "content": "hi"}])
        finally:
            retry_cfg.wait, retry_cfg.stop = old_wait, old_stop

        assert resp.content == '{"answer": 7}'
        assert len(calls) == 2


def _make_chunk(content=None, reasoning=None, finish=None, tool_calls=None,
                usage=None, empty_choices=False):
    """构造流式 chunk 替身。empty_choices=True 模拟仅含 usage 的收尾 chunk。"""
    if empty_choices:
        return SimpleNamespace(choices=[], usage=usage)
    delta = SimpleNamespace(
        content=content, reasoning_content=reasoning, tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_tc_delta(index, tc_id=None, name=None, arguments=None):
    """构造流式 tool_calls 分片。"""
    return SimpleNamespace(
        index=index, id=tc_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _MockStreamClient:
    """create() 返回 chunk 迭代器的同步替身。"""

    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self._chunks)


class _MockAsyncStreamClient:
    """create() 返回异步 chunk 迭代器的替身。"""

    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    async def _create(self, **kwargs):
        self.calls.append(kwargs)
        chunks = list(self._chunks)

        async def _gen():
            for c in chunks:
                yield c

        return _gen()


class TestStreamMode:
    """stream=True 流式调用（DashScope 思考模式对非流式强制 400）。

    流式响应必须拼装成与非流式一致的形状：content/reasoning_content
    拼接、tool_calls 分片重组、usage 从收尾 chunk 提取。
    """

    def test_stream_collects_content_reasoning_usage(self):
        adapter = OpenAICompatAdapter(
            model="qwen3-32b",
            api_key="sk-test",
            base_url="http://localhost:9999",
            stream=True,
        )
        client = _MockStreamClient([
            _make_chunk(reasoning="先想"),
            _make_chunk(reasoning="一下"),
            _make_chunk(content='{"answer": '),
            _make_chunk(content='1}', finish="stop"),
            _make_chunk(empty_choices=True,
                        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7)),
        ])
        adapter._get_sync_client = lambda: client

        resp = adapter.call([{"role": "user", "content": "hi"}])

        assert resp.content == '{"answer": 1}'
        assert resp.usage.prompt_tokens == 5
        assert resp.usage.completion_tokens == 7
        assert client.calls[0]["stream"] is True
        assert client.calls[0]["stream_options"] == {"include_usage": True}

    def test_stream_assembles_tool_call_fragments(self):
        adapter = OpenAICompatAdapter(
            model="qwen3-32b",
            api_key="sk-test",
            base_url="http://localhost:9999",
            stream=True,
        )
        client = _MockStreamClient([
            _make_chunk(tool_calls=[_make_tc_delta(0, tc_id="call_1", name="run_python")]),
            _make_chunk(tool_calls=[_make_tc_delta(0, arguments='{"code": "pri')]),
            _make_chunk(tool_calls=[_make_tc_delta(0, arguments='nt(1)"}')],
                        finish="tool_calls"),
        ])
        adapter._get_sync_client = lambda: client

        resp = adapter.call([{"role": "user", "content": "hi"}])

        assert resp.tool_calls == [{
            "id": "call_1",
            "name": "run_python",
            "arguments": '{"code": "print(1)"}',
        }]

    @pytest.mark.asyncio
    async def test_agen_stream_collects_content(self):
        adapter = OpenAICompatAdapter(
            model="qwen3-32b",
            api_key="sk-test",
            base_url="http://localhost:9999",
            stream=True,
        )
        client = _MockAsyncStreamClient([
            _make_chunk(reasoning="嗯"),
            _make_chunk(content='{"answer": 2}', finish="stop"),
            _make_chunk(empty_choices=True,
                        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4)),
        ])
        adapter._get_client = lambda: client

        resp = await adapter.agen([{"role": "user", "content": "hi"}])

        assert resp.content == '{"answer": 2}'
        assert resp.usage.completion_tokens == 4
        assert client.calls[0]["stream"] is True

    def test_stream_off_by_default(self):
        adapter = OpenAICompatAdapter(
            model="test-model",
            api_key="sk-test",
            base_url="http://localhost:9999",
        )
        ok = _make_response('{"answer": 1}', "stop")
        client = _MockSyncClient([ok])
        adapter._get_sync_client = lambda: client

        adapter.call([{"role": "user", "content": "hi"}])

        assert "stream" not in client.calls[0]


def _make_bad_request(msg: str):
    """构造 openai.BadRequestError 替身（真实异常类型，消息自定义）。"""
    import httpx
    from openai import BadRequestError

    return BadRequestError(
        msg,
        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
        body=None,
    )


class TestDashScopeMaxTokensClamp:
    """DashScope 风格 max_tokens 上限错误的自动降配。

    百炼报错形如 "Range of max_tokens should be [1, 16384]"（completion
    上限，非总上下文），旧解析只认 vLLM 的 max_model_len 格式，无法降配，
    导致 400 重试耗尽。
    """

    def test_clamp_recognizes_dashscope_range_error(self):
        adapter = OpenAICompatAdapter(
            model="qwen3-32b",
            api_key="sk-test",
            base_url="http://localhost:9999",
        )
        err = _make_bad_request(
            "<400> InternalError.Algo.InvalidParameter: "
            "Range of max_tokens should be [1, 16384]"
        )
        assert adapter._clamp_max_tokens_for_bad_request(err, 32000) == 16384

    def test_clamp_recognizes_dashscope_range_error_from_api_error(self):
        """流式模式下该错误以 openai.APIError（非 BadRequestError）抛出。"""
        import httpx
        from openai import APIError

        adapter = OpenAICompatAdapter(
            model="qwen3-32b",
            api_key="sk-test",
            base_url="http://localhost:9999",
        )
        err = APIError(
            "<400> InternalError.Algo.InvalidParameter: "
            "Range of max_tokens should be [1, 16384]",
            request=httpx.Request("POST", "http://x"),
            body=None,
        )
        assert adapter._clamp_max_tokens_for_bad_request(err, 32000) == 16384

    def test_call_recovers_after_dashscope_clamp(self):
        adapter = OpenAICompatAdapter(
            model="qwen3-32b",
            api_key="sk-test",
            base_url="http://localhost:9999",
            max_tokens=32000,
        )
        err = _make_bad_request(
            "<400> InternalError.Algo.InvalidParameter: "
            "Range of max_tokens should be [1, 16384]"
        )
        ok = _make_response('{"answer": 1}', "stop")
        client = _MockSyncClient([err, ok])
        # 让第一次 create 抛错而非返回
        responses = [err, ok]

        def create(**kwargs):
            client.calls.append(kwargs)
            r = responses.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        client.chat.completions.create = create
        adapter._get_sync_client = lambda: client

        resp = adapter.call([{"role": "user", "content": "hi"}])

        assert resp.content == '{"answer": 1}'
        assert client.calls[0]["max_tokens"] == 32000
        assert client.calls[1]["max_tokens"] == 16384
