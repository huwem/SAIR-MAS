"""Agent 基类和通用 LLM Agent 实现。"""

import asyncio
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo._registries import tool_registry
from mas_topo.tools.builtin import RUN_PYTHON_TOOL_NAME

logger = logging.getLogger(__name__)

# 工具调用最大轮数（避免无限循环）
MAX_TOOL_ROUNDS = 3


class BaseAgent(ABC):
    """Agent 抽象基类。

    定义 execute/async_execute 接口和记忆管理。
    """

    def __init__(
        self,
        name: str = "",
        role: str = "",
        llm_adapter: Optional[Any] = None,
        tools: Optional[list] = None,
        memory_limit: int = 3000,
        memory_max_rounds: Optional[int] = None,
        local_neighbor_table: Optional[dict[str, set[str]]] = None,
    ):
        self.name = name
        self.role = role
        self.llm_adapter = llm_adapter
        self.tools = tools or []
        self.memory_limit = memory_limit
        self.memory_max_rounds = memory_max_rounds
        # 统一局部邻居表：key 为 Agent 名称，value 为该 Agent 的邻居集合。
        # 其中 key == self.name 的条目表示当前 Agent 的物理一跳邻居（由框架/外部拓扑维护，只读）。
        # 其他条目为 Agent 通过交互学习到的网络认知（可通过 update_topology_view 工具更新）。
        self.local_neighbor_table: dict[str, set[str]] = local_neighbor_table or {}
        self._memory_history: list[str] = []
        self._memory_rounds: list[int] = []  # 与 _memory_history 平行的轮次记录

    def get_memory_entries(self) -> list[str]:
        """获取过滤后的记忆条目列表（与 get_memory_context 相同的窗口规则）。"""
        if not self._memory_history:
            return []
        if self.memory_max_rounds is not None and self.memory_max_rounds > 0 and self._memory_rounds:
            current_round = max(self._memory_rounds)
            min_round = current_round - self.memory_max_rounds + 1
            return [
                content for content, rnd in zip(self._memory_history, self._memory_rounds)
                if rnd >= min_round
            ]
        return list(self._memory_history)

    def get_memory_context(self) -> str:
        """获取记忆上下文文本。

        当 memory_max_rounds 启用时，仅保留最近 N 轮的记忆条目。
        """
        entries = self.get_memory_entries()
        if not entries:
            return "No prior context."
        return "\n\n".join(entries)

    def append_memory(self, content: str, round_num: int = 0) -> None:
        """追加记忆（memory_limit > 0 时截断到该长度，0/None 不截断；记录轮次用于窗口过滤）。"""
        if self.memory_limit and self.memory_limit > 0:
            content = content[:self.memory_limit]
        self._memory_history.append(content)
        self._memory_rounds.append(round_num)

    def clear_memory(self) -> None:
        """清空记忆。"""
        self._memory_history.clear()
        self._memory_rounds.clear()

    @abstractmethod
    def execute(self, input_data: Any, context: ExecutionContext, **kwargs) -> AgentOutput:
        """同步执行。"""

    @abstractmethod
    async def async_execute(
        self, input_data: Any, context: ExecutionContext, **kwargs
    ) -> AgentOutput:
        """异步执行。"""


class LLMAgent(BaseAgent):
    """通用 LLM 驱动的 Agent。

    system_prompt、output_format、output_schema 均可配置注入，
    替代硬编码的各类 Agent 子类。
    """

    # 字段名 → AgentOutput 属性 getter 映射
    _FIELD_GETTERS = {
        "public_content": lambda o: o.public_content,
        "q_vector": lambda o: o.query_descriptor,
        "k_vector": lambda o: o.key_descriptor,
        "q_desc": lambda o: o.query_descriptor,
        "k_desc": lambda o: o.key_descriptor,
        "messages": lambda o: o.messages,
        "answer": lambda o: o.answer,
        "is_complete": lambda o: o.is_complete,
        "next_goal": lambda o: o.next_goal,
        "next_speaker": lambda o: o.next_speaker,
        "private_content": lambda o: o.private_content,
        "tool_call": lambda o: o.tool_call,
        # --- PragmaMAS-P2P fields ---
        "private_routing": lambda o: o.private_routing,
        "vote_complete": lambda o: o.vote_complete,
    }

    def __init__(
        self,
        name: str = "",
        role: str = "",
        llm_adapter: Optional[Any] = None,
        system_prompt: Optional[str] = None,
        output_format: str = "text",          # "text" | "json"
        output_schema: Optional[dict] = None,  # JSON 输出字段定义
        tools: Optional[list] = None,
        memory_limit: int = 3000,
        required_output_fields: Optional[list[str]] = None,
        context_fields: Optional[list[str]] = None,
        bare_io: bool = False,
        local_neighbor_table: Optional[dict[str, set[str]]] = None,
        memory_max_rounds: Optional[int] = None,
        max_tool_rounds: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            name=name, role=role,
            llm_adapter=llm_adapter, tools=tools,
            memory_limit=memory_limit,
            memory_max_rounds=memory_max_rounds,
            local_neighbor_table=local_neighbor_table,
        )
        self._system_prompt = system_prompt
        self.output_format = output_format
        self.output_schema = output_schema
        self.required_output_fields = list(required_output_fields) if required_output_fields else []
        self.context_fields = list(context_fields) if context_fields else []
        # bare_io=True：裸调用模式——无 system prompt，user prompt 即完整题目原文，
        # 输出全文直接作为 answer（用于 direct 单智能体基线）
        self.bare_io = bare_io
        self.max_tool_rounds = max_tool_rounds

    @property
    def system_prompt(self) -> str:
        if self._system_prompt:
            return self._system_prompt
        return f"You are {self.name}, the {self.role}."

    def _derive_output_schema(self) -> dict:
        """根据 required_output_fields 推导最小 JSON Schema。"""
        schema: dict = {}
        for field in self.required_output_fields:
            if field == "private_routing":
                schema[field] = [{
                    "recipients": ["string"],
                    "illocution": "string",
                    "content": "string",
                    "answer_correct": "boolean",
                }]
            elif field == "messages":
                schema[field] = [{"to": "string", "content": "string"}]
            elif field == "tool_call":
                schema[field] = {"name": "string", "arguments": {}}
            elif field in ("vote_complete", "is_complete"):
                schema[field] = "boolean"
            elif field in ("answer", "public_content", "next_goal", "next_speaker", "query_descriptor", "key_descriptor", "q_desc", "k_desc"):
                schema[field] = "string"
            elif field == "private_content":
                schema[field] = {"illocution[Category]": "content"}
            else:
                schema[field] = "any"
        return schema

    def _build_prompt(
        self,
        input_data: Any,
        context: ExecutionContext,
    ) -> tuple[str, str]:
        """构建 system_prompt 和 user_prompt。"""
        # bare_io 模式：无 system prompt，user prompt 就是题目原文，不注入任何其他内容
        if self.bare_io:
            return "", str(context.task or "")

        system = self.system_prompt

        _system_lower = system.lower()
        _has_json_format = "strict response format" in _system_lower
        # 仅 P2P 方法（使用 private_routing）需要自动注入队友/拓扑信息
        _is_p2p = "private_routing" in self.required_output_fields
        user_parts = []
        if context.task:
            user_parts.append(f"Task: {context.task}")

        # 注入当前轮次进度和本轮目标
        if context.round_num > 0:
            progress = f"Round {context.round_num}. {context.goal}"
            user_parts.append(progress)

        # P2P 方法：注入 team roster 和拓扑视图，帮助生成正确的 private_routing
        if _is_p2p:
            if context.team_roster:
                other_agents = [
                    a for a in context.team_roster
                    if a["name"] != self.name
                ]
                if other_agents:
                    roster_parts = [a["name"] for a in other_agents]
                    roster_line = f"Team: {', '.join(roster_parts)}"
                    user_parts.append(roster_line)

            all_agent_names = {a["name"] for a in context.team_roster} if context.team_roster else set()
            if not all_agent_names and context.neighbor_map:
                all_agent_names = set(context.neighbor_map.keys())

            my_neighbors = sorted(self.local_neighbor_table.get(self.name, set()))
            non_neighbors = sorted(all_agent_names - set(my_neighbors) - {self.name})

            is_neighbor_topology = context.topology_strategy == "neighbor"
            # 仅当存在非邻居（拓扑受限，如 chain）时注入邻居表和可达性提示；
            # 全连通（full）下所有 agent 互为邻居，注入只会误导和干扰。
            if non_neighbors:
                neighbor_names = ", ".join(my_neighbors) if my_neighbors else "none"
                user_parts.append(f"Neighbors: {neighbor_names}")
                if is_neighbor_topology and my_neighbors:
                    user_parts.append(
                        "Only your neighbors are routable: messages addressed to "
                        "non-neighbors will NOT be delivered. "
                        "You may ask a neighbor to relay messages to non-neighbors."
                    )

        memory_ctx = self.get_memory_context()
        if memory_ctx != "No prior context.":
            user_parts.append(f"Your memory:\n{memory_ctx}")

        # 注入当前答案（与记忆分离，独立展示）。
        # 若 Agent 已通过 context_fields 显式声明 current_answer 则由
        # context_fields 循环负责注入，避免同一内容出现两次。
        meta = getattr(context, "metadata", None) or {}
        answer_source = str(meta.get("current_answer_source") or "").lower()
        if "current_answer" not in self.context_fields:
            current_answer = meta.get("current_answer")
            if current_answer:
                answer_status = meta.get("current_answer_status", "")
                status_note = f" (status: {answer_status})" if answer_status else ""
                # 标注答案来源角色，避免把某方 answer 误当作团队共识事实
                title = f"Answer from {answer_source}" if answer_source else "Current answer"
                user_parts.append(f"{title}{status_note}:\n{current_answer[:3000]}")

        # 按 Agent 配置注入上下文中的额外字段（跳过 None 和空字符串，
        # 避免出现 "[current_answer]\n\n" 这类空标题占位）
        for key in self.context_fields:
            value = meta.get(key)
            if value is not None and value != "":
                # current_answer 的标签改写为 answer_from_<来源角色>
                label = key
                if key == "current_answer" and answer_source:
                    label = f"answer_from_{answer_source}"
                user_parts.append(f"[{label}]\n{value}")

        # 注入 per-agent ledger 摘要（当前状态视图，不存记忆）
        ledger_summaries = getattr(context, "metadata", {}).get("__ledger_summaries__", {})
        if isinstance(ledger_summaries, dict):
            my_ledger_summary = ledger_summaries.get(self.name, "")
            if my_ledger_summary:
                user_parts.append(my_ledger_summary)

        # 论文格式时在 user prompt 末尾追加格式提醒
        if _has_json_format:
            user_parts.append("Remember: output ONLY the JSON object, no other text.")

        user = "\n\n".join(user_parts)
        return system, user

    @staticmethod
    def _chat_messages(system_prompt: str, user_content: str) -> list[dict]:
        """组装 chat messages；system 为空（bare_io）时省略 system 消息。"""
        if system_prompt:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        return [{"role": "user", "content": user_content}]

    def _build_tool_schemas(self) -> list[dict]:
        """将 self.tools 转换为 OpenAI 原生 tool calling 格式。"""
        schemas = []
        for tool in self.tools:
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description[:1024],
                    "parameters": tool.parameters,
                },
            }
            schemas.append(schema)
        return schemas

    def _parse_response(self, response: str) -> AgentOutput:
        """按 output_format 解析 LLM 响应为 AgentOutput。"""
        # bare_io 模式：输出全文直接作为 answer，不做任何结构化解析
        if self.bare_io:
            text = response.strip()
            return AgentOutput(answer=text, public_content=text, raw_response=response)
        if self.output_format == "json":
            output = self._parse_json_response(response)
            return self._apply_output_defaults(output)
        return AgentOutput.from_text(response)

    @staticmethod
    def _apply_output_defaults(output: AgentOutput) -> AgentOutput:
        """为可选字段填充合理默认值，减少因 LLM 遗漏字段导致的重试。"""
        if output.vote_complete is None:
            output.vote_complete = False
        if output.private_routing is None:
            output.private_routing = []
        if output.tool_calls is None:
            output.tool_calls = []
        return output

    def _enforce_tool_result_honesty(self, output: AgentOutput) -> AgentOutput:
        """如果 Agent 本轮的 run_python 工具有任何一次失败，强制 vote_complete=false。

        这是框架层硬性约束：LLM 可能产生“测试全部通过”的幻觉，但工具客观结果优先。
        只有所有 run_python 调用都通过（或没有调用 run_python）时，才保留 LLM 原来的 vote_complete。
        """
        tool_calls = getattr(output, "tool_calls", None) or []
        has_failed_test = False
        for call_info in tool_calls:
            if not isinstance(call_info, dict):
                continue
            call = call_info.get("call", {}) or {}
            result = call_info.get("result", {}) or {}
            if call.get("name") != RUN_PYTHON_TOOL_NAME:
                continue
            result_data = result.get("result") if isinstance(result, dict) else result
            if isinstance(result_data, dict) and result_data.get("passed") is False:
                has_failed_test = True
                break

        if has_failed_test and output.vote_complete:
            logger.warning(
                "Agent %s: at least one run_python call failed; forcing vote_complete from true to false.",
                self.name,
            )
            output.vote_complete = False
            # 在 public_content 里追加系统通知，让其他 Agent 在记忆/路由中能看到 Tester 被强制否决
            notice = " [SYSTEM NOTICE: answer_correct was forced to false because a run_python call failed.]"
            output.public_content = (output.public_content or "") + notice

        return output

    def _parse_json_response(self, response: str) -> AgentOutput:
        """解析 JSON 格式的 LLM 响应。

        支持论文字段名 (public_content/q_vector 等) 和旧字段名
        (publiccontent/qvector 等) 两种格式。
        使用 Pydantic 验证字段类型和必填项。
        """
        from mas_topo.message import AgentOutputSchema

        parsed = self._extract_json(response)
        if parsed is not None:
            # 容错：LLM 偶尔会输出 JSON 数组，取第一个对象或降级为空对象
            if isinstance(parsed, list):
                parsed = next((x for x in parsed if isinstance(x, dict)), {})
            if not isinstance(parsed, dict):
                parsed = {}

            # 字段别名：同时支持两种命名约定
            def _get(*keys):
                for k in keys:
                    val = parsed.get(k)
                    if val is not None:
                        return val
                return None

            # 通过 Pydantic 验证 + 类型容错（兼容空格/ snake_case 字段名）
            try:
                validated = AgentOutputSchema.model_validate(parsed)

                # 如果顶层没有投票字段，从 private_routing 条目的
                # answer_correct（新）/ vote_complete（旧）推导
                vote_complete = validated.vote_complete
                if vote_complete is None and validated.private_routing:
                    routing_votes = [
                        bool(item.get("answer_correct", item.get("vote_complete")))
                        for item in validated.private_routing
                        if isinstance(item, dict)
                        and ("answer_correct" in item or "vote_complete" in item)
                    ]
                    if routing_votes:
                        if len(set(routing_votes)) > 1:
                            logger.warning(
                                "Agent %s: inconsistent vote_complete across routing items, defaulting to false",
                                self.name,
                            )
                            vote_complete = False
                        else:
                            vote_complete = routing_votes[0]

                return AgentOutput(
                    public_content=validated.public_content,
                    private_content=validated.private_content,
                    messages=validated.messages,
                    query_descriptor=validated.query_descriptor,
                    key_descriptor=validated.key_descriptor,
                    is_complete=validated.is_complete,
                    next_goal=validated.next_goal,
                    next_speaker=validated.next_speaker,
                    answer=validated.answer,
                    tool_call=validated.tool_call,
                    private_routing=validated.private_routing,
                    vote_complete=vote_complete,
                    raw_response=response,
                )
            except Exception:
                pass

            # Pydantic 验证失败时回退到手动字段提取（保留旧逻辑作为安全网）
            priv = _get("private_content", "privatecontent", "private content")
            if isinstance(priv, dict):
                normalized = {}
                for k, v in priv.items():
                    if isinstance(v, str):
                        normalized[k] = v
                    elif isinstance(v, dict):
                        normalized[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        normalized[k] = str(v)
                priv = normalized
            else:
                priv = {}

            msgs = _get("messages")
            if isinstance(msgs, list):
                validated_msgs = []
                for m in msgs:
                    if isinstance(m, dict) and "to" in m and "content" in m:
                        validated_msgs.append({
                            "to": str(m["to"]),
                            "content": str(m["content"]),
                        })
                msgs = validated_msgs
            else:
                msgs = []

            def _str_or_none(v):
                return None if v is None else str(v)

            raw_routing = _get("private_routing", "privaterouting")
            if isinstance(raw_routing, list):
                routing = [item for item in raw_routing if isinstance(item, dict)]
            else:
                routing = []

            return AgentOutput(
                public_content=_get("public_content", "publiccontent", "public content") or "",
                private_content=priv,
                messages=msgs,
                query_descriptor=_str_or_none(_get("q_vector", "qvector", "q_desc", "q desc")),
                key_descriptor=_str_or_none(_get("k_vector", "kvector", "k_desc", "k desc")),
                is_complete=_get("is_complete", "iscomplete", "is complete"),
                next_goal=_str_or_none(_get("next_goal", "nextgoal", "next goal")),
                next_speaker=_str_or_none(_get("next_speaker", "nextspeaker", "next speaker")),
                answer=_str_or_none(_get("answer", "Answer", "ANSWER", "final_answer")),
                tool_call=_get("tool_call"),
                private_routing=routing,
                vote_complete=_get("answer_correct", "answercorrect", "answer correct", "vote_complete", "votecomplete"),
                raw_response=response,
            )

        # 兜底：成熟库解析失败，记录原始响应并返回结构化输出
        preview = response[:200].replace('\n', ' ') if response else "<empty response>"
        logger.warning(
            "Agent %s: JSON parse failed, returning raw response. "
            "Preview: %s",
            self.name, preview,
        )
        public_content = response if response else "[Empty or unparseable LLM response]"
        return AgentOutput(
            public_content=public_content[:500],
            raw_response=response,
        )

    @staticmethod
    def _first_balanced_braces(text: str) -> Optional[str]:
        """从文本中提取第一个平衡的 {...} 片段，忽略字符串内的花括号。"""
        start = text.find('{')
        if start < 0:
            return None
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == '\\':
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if not in_str:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        return text[start:i + 1]
        return None

    @staticmethod
    def _extract_json(response: str) -> Optional[dict]:
        """从 LLM 响应中提取 JSON 对象，多层回退策略。

        第 1 层：标准 json.loads（应对格式良好的 JSON）
        第 2 层：json_repair 直接修复全文
        第 3 层：从 markdown 代码块提取后 json_repair / json.loads
        第 4 层：从第一个平衡花括号片段提取后 json_repair / json.loads
        """
        # 优先尝试标准库，速度最快且不需要 json_repair
        try:
            stripped = response.strip()
            return json.loads(stripped)
        except Exception:
            pass

        # 尝试导入 json_repair；若未安装，后续层仅使用标准 json.loads
        try:
            from json_repair import repair_json
        except Exception:
            repair_json = None  # type: ignore[assignment]

        def _loads(text: str) -> Optional[dict]:
            try:
                return json.loads(text)
            except Exception:
                if repair_json is not None:
                    try:
                        repaired = repair_json(text)
                        if repaired is not None:
                            return json.loads(repaired)
                    except Exception:
                        pass
                return None

        # 第 1 层：全文修复
        result = _loads(response)
        if result is not None:
            return result

        # 第 2 层：从 markdown 代码块提取
        md_match = re.search(
            r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL,
        )
        if md_match:
            result = _loads(md_match.group(1))
            if result is not None:
                return result

        # 第 3 层：定位第一个平衡花括号片段
        balanced = LLMAgent._first_balanced_braces(response)
        if balanced:
            result = _loads(balanced)
            if result is not None:
                return result

        return None

    # ── 工具执行支持 ────────────────────────────────────────

    # ── 输出格式验证与重试 ────────────────────────────────

    def _validate_output(self, output: AgentOutput) -> list[str]:
        """验证 Agent 输出是否包含配置的必填字段。

        字段名通过 required_output_fields 配置（默认空列表，不强制验证）。
        支持标准字段名: public_content, q_vector, k_vector, messages, answer 等。

        验证规则:
        - 字符串字段（public_content, q_vector, k_vector, answer, next_goal）: 必须非空且非空白。
        - 列表/字典字段（messages, private_content）: 只要存在即可，空值合法。
        - 布尔字段（is_complete）: 只要存在即可（True/False 都合法）。

        Returns:
            缺失或为空的字段名列表。空列表表示验证通过。
        """
        missing = []
        for field_name in self.required_output_fields:
            getter = self._FIELD_GETTERS.get(field_name)
            if getter is None:
                continue
            value = getter(output)
            if value is None:
                missing.append(field_name)
                continue
            if isinstance(value, str) and not value.strip():
                missing.append(field_name)
        return missing

    def _run_tool(self, tool_name: str, arguments: dict) -> dict:
        """执行单个工具调用。

        Args:
            tool_name: 工具注册名。
            arguments: 工具参数字典。

        Returns:
            {"success": bool, "result": Any, "error": str | None}
        """
        if not tool_registry.has(tool_name):
            return {
                "success": False,
                "result": None,
                "error": f"Unknown tool: {tool_name}",
            }

        try:
            tool = tool_registry.get(tool_name)
            # 将当前 Agent 实例传入工具，使工具可访问 Agent 状态（如 local_topology_view）
            result = tool.run(agent=self, **arguments)
            return {"success": True, "result": result, "error": None}
        except TypeError as e:
            return {
                "success": False,
                "result": None,
                "error": f"Tool argument error: {e}",
            }
        except Exception as e:
            return {
                "success": False,
                "result": None,
                "error": f"Tool execution failed: {e}",
            }

    @staticmethod
    def _format_tool_result(tool_result: dict) -> str:
        """将工具结果格式化为简洁的 LLM 友好摘要。

        对于 run_python 工具，提取 passed/stdout/stderr 关键信息；
        stdout 超限时截断并显式标注（避免"PASSED 但看不到完整输出"的错觉），
        stderr traceback 保留首尾各 400 字符。
        """
        result = tool_result.get("result") or tool_result
        if not isinstance(result, dict):
            return json.dumps(tool_result, ensure_ascii=False)

        passed = result.get("passed")
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        returncode = result.get("returncode")

        parts = []
        if passed is not None:
            parts.append(f"PASSED: {passed}")
        if returncode is not None:
            parts.append(f"returncode: {returncode}")
        if stdout:
            if len(stdout) > 500:
                stdout = stdout[:500] + f"\n...(truncated, {len(stdout)} chars total)"
            parts.append(f"stdout: {stdout}")
        elif passed:
            # PASSED 但零输出：代码可能未真正产出结果（或 print 丢失），
            # 显式提示，防止模型把"跑通了"误读为"验证通过"。
            parts.append("stdout: <empty — code produced no output>")
        if stderr:
            if len(stderr) > 800:
                stderr = stderr[:400] + "\n...(truncated)...\n" + stderr[-400:]
            parts.append(f"stderr: {stderr}")

        return "\n".join(parts)

    @staticmethod
    def _format_tool_input(call: dict) -> str:
        """将工具调用的输入参数完整格式化（不截断——模型需要看到自己的完整代码）。"""
        arguments = call.get("arguments", {})
        try:
            rendered = json.dumps(arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            rendered = str(arguments)
        return rendered

    def _build_tool_summary(self, executed_calls: list[dict]) -> str:
        """把已执行工具调用汇总为紧凑文本摘要（含输入与输出）。

        用于最终 JSON 输出轮次，替代冗长的 Full Message History。
        """
        if not executed_calls:
            return ""
        parts = []
        for call_info in executed_calls:
            call = call_info.get("call", {}) or {}
            result = call_info.get("result", {}) or {}
            tool_name = call.get("name", "")
            parts.append(
                f"Tool '{tool_name}' input:\n{self._format_tool_input(call)}\n"
                f"Tool '{tool_name}' result:\n{self._format_tool_result(result)}"
            )
        return "\n\n".join(parts)

    def _build_tool_followup_content(
        self,
        context,
        user_prompt: str,
        summary: str,
        final: bool,
    ) -> str:
        """构造工具调用后的续轮/收尾 user prompt。

        显式声明"工具不是 team member"并列出合法收件人，避免模型把
        run_python 等工具名误写进 private_routing 的 recipients。
        """
        if final:
            closing = (
                "Now output your final response as a valid JSON object "
                "matching the required schema."
            )
        else:
            closing = (
                "Based on the tool result above, either make another tool call or "
                "output your final response as a valid JSON object matching the required schema."
            )
        note = "Tools are NOT team members."
        teammates = [
            a["name"] for a in (getattr(context, "team_roster", None) or [])
            if a["name"] != self.name
        ]
        if teammates:
            note += (
                f" 'recipients' must be team member names only ({', '.join(teammates)}); "
                "never address messages to tools."
            )
        return (
            f"{user_prompt}\n\n{summary}\n\n"
            f"{closing} {note} Output ONLY the JSON, no markdown, no explanations."
        )

    def _process_native_tool_calls(
        self,
        tool_calls: list[dict],
        executed_calls: list[dict],
    ) -> str:
        """执行 OpenAI 原生 tool_calls 并返回紧凑结果摘要。

        不再把完整 assistant/tool 消息对追加到 messages，避免 Full Message History
        过长。调用方负责把返回的摘要组装进 prompt。
        """
        summaries = []
        for tc in tool_calls:
            tool_name = tc["name"]
            try:
                arguments = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
            except json.JSONDecodeError:
                arguments = {}

            logger.info(
                "Agent %s: calling tool '%s' with args=%s",
                self.name, tool_name, json.dumps(arguments, ensure_ascii=False)[:200],
            )
            tool_result = self._run_tool(tool_name, arguments)
            logger.info(
                "Agent %s: tool '%s' result: %s",
                self.name, tool_name,
                json.dumps(tool_result, ensure_ascii=False)[:300],
            )
            executed_calls.append({"call": {"name": tool_name, "arguments": arguments}, "result": tool_result})

            tool_summary = self._format_tool_result(tool_result)
            result_content = (
                f"Tool '{tool_name}' input:\n{self._format_tool_input({'arguments': arguments})}\n"
                f"Tool '{tool_name}' result:\n{tool_summary}\n"
            )
            if tool_name == RUN_PYTHON_TOOL_NAME:
                result_data = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else tool_result
                passed = result_data.get("passed") if isinstance(result_data, dict) else None
                if passed is False or not passed:
                    result_content += (
                        "CRITICAL for run_python: if the test FAILED, you MUST NOT set answer_correct=true. "
                        "If the failure is due to malformed arguments (e.g., SyntaxError, ImportError, "
                        "or a truncated function signature), you MUST fix the arguments and call run_python again. "
                        "Only claim tests passed when the tool result shows PASSED: True."
                    )
            else:
                result_content += "IMPORTANT: if the tool call FAILED, report the failure in your response."
            summaries.append(result_content)

        return "\n".join(summaries)

    async def _process_native_tool_calls_async(
        self,
        tool_calls: list[dict],
        executed_calls: list[dict],
    ) -> str:
        """异步执行 OpenAI 原生 tool_calls 并返回紧凑结果摘要。"""
        summaries = []
        for tc in tool_calls:
            tool_name = tc["name"]
            try:
                arguments = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
            except json.JSONDecodeError:
                arguments = {}

            logger.info(
                "Agent %s: calling tool '%s' with args=%s",
                self.name, tool_name, json.dumps(arguments, ensure_ascii=False)[:200],
            )
            # 工具执行（如 run_python 的 subprocess.run）是同步阻塞调用，
            # 若在事件循环线程上直接运行会阻塞整个循环，进而饿死 LLM 请求
            # 的墙钟截止时间轮询。放到线程执行器里，事件循环保持可调度。
            loop = asyncio.get_running_loop()
            tool_result = await loop.run_in_executor(
                None, self._run_tool, tool_name, arguments
            )
            logger.info(
                "Agent %s: tool '%s' result: %s",
                self.name, tool_name,
                json.dumps(tool_result, ensure_ascii=False)[:300],
            )
            executed_calls.append({"call": {"name": tool_name, "arguments": arguments}, "result": tool_result})

            tool_summary = self._format_tool_result(tool_result)
            result_content = (
                f"Tool '{tool_name}' input:\n{self._format_tool_input({'arguments': arguments})}\n"
                f"Tool '{tool_name}' result:\n{tool_summary}\n"
            )
            if tool_name == RUN_PYTHON_TOOL_NAME:
                result_data = tool_result.get("result") if isinstance(tool_result.get("result"), dict) else tool_result
                passed = result_data.get("passed") if isinstance(result_data, dict) else None
                if passed is False or not passed:
                    result_content += (
                        "CRITICAL for run_python: if the test FAILED, you MUST NOT set answer_correct=true. "
                        "If the failure is due to malformed arguments (e.g., SyntaxError, ImportError, "
                        "or a truncated function signature), you MUST fix the arguments and call run_python again. "
                        "Only claim tests passed when the tool result shows PASSED: True."
                    )
            else:
                result_content += "IMPORTANT: if the tool call FAILED, report the failure in your response."
            summaries.append(result_content)

        return "\n".join(summaries)

    # ── 执行入口 ────────────────────────────────────────────

    def execute(self, input_data: Any, context: ExecutionContext, **kwargs) -> AgentOutput:
        t_start = time.time()
        max_retries = kwargs.get("max_retries", 2)
        system_prompt, user_prompt = self._build_prompt(input_data, context)

        messages = self._chat_messages(system_prompt, user_prompt)

        if self.llm_adapter is None:
            logger.warning("Agent %s: no LLM adapter configured", self.name)
            return AgentOutput(
                public_content="[No LLM adapter configured]",
                raw_response="",
            )

        output = None
        last_response = ""
        last_usage = None
        executed_calls: list[dict] = []
        tool_schemas = self._build_tool_schemas() if self.tools else None

        for attempt in range(max_retries + 1):
            executed_calls = []  # 每轮重试隔离工具调用记录
            # 工具调用子循环：LLM 可通过原生 tool_calls 多次调用工具
            for tool_round in range(self.max_tool_rounds or MAX_TOOL_ROUNDS):
                logger.info(
                    "Agent %s (%s): calling LLM (attempt %d/%d, tool_round %d)...",
                    self.name, self.role, attempt + 1, max_retries + 1, tool_round + 1,
                )
                llm_response = self.llm_adapter.call(
                    messages,
                    tools=tool_schemas,
                )
                last_response = llm_response.content or ""
                last_usage = llm_response.usage

                if llm_response.tool_calls:
                    tool_summary = self._process_native_tool_calls(llm_response.tool_calls, executed_calls)
                    # 重置 messages：智能体只看当前一次工具调用结果，不累积 Full Message History
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self._build_tool_followup_content(
                            context, user_prompt, tool_summary, final=False,
                        )},
                    ]
                    continue

                # 无 tool_calls：如果之前调用过工具，需要一次 JSON mode 收尾，
                # 确保“正常输出”符合系统提示词的 JSON 要求（tool calling 阶段禁用 JSON mode）
                if executed_calls:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self._build_tool_followup_content(
                            context, user_prompt, self._build_tool_summary(executed_calls),
                            final=True,
                        )},
                    ]
                    logger.info(
                        "Agent %s: making final JSON-mode call after %d tool call(s)",
                        self.name, len(executed_calls),
                    )
                    llm_response = self.llm_adapter.call(
                        messages,
                        json_mode=True,
                    )
                    last_response = llm_response.content or ""
                    last_usage = llm_response.usage

                # 解析输出
                output = self._parse_response(last_response)
                output.tool_calls = executed_calls
                break
            else:
                # tool 循环耗尽（MAX_TOOL_ROUNDS 次全是 tool_calls）
                if executed_calls:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self._build_tool_followup_content(
                            context, user_prompt, self._build_tool_summary(executed_calls),
                            final=True,
                        )},
                    ]
                    llm_response = self.llm_adapter.call(
                        messages,
                        json_mode=True,
                    )
                    last_response = llm_response.content or ""
                    last_usage = llm_response.usage

                if last_response:
                    output = self._parse_response(last_response)
                    output.tool_calls = executed_calls
                else:
                    output = AgentOutput(
                        public_content="[Tool loop exhausted without final output]",
                        raw_response="",
                        tool_calls=executed_calls,
                    )

            missing = self._validate_output(output)
            if not missing:
                break

            if attempt < max_retries:
                logger.warning(
                    "Agent %s: missing fields %s, retrying...",
                    self.name, missing,
                )
                if not last_response or not last_response.strip():
                    feedback = (
                        "Your previous response was empty. "
                        "Please output a valid JSON object with ALL required fields. "
                        "Output ONLY the JSON, no markdown, no explanations."
                    )
                else:
                    missing_fields_str = ", ".join(missing)
                    feedback = (
                        f"Your previous response was invalid: "
                        f"missing required fields [{missing_fields_str}]. "
                        f"Please output a valid JSON object with ALL required fields. "
                        f"Output ONLY the JSON, no markdown, no explanations."
                    )
                messages.append({"role": "user", "content": feedback})

        if missing:
            logger.error(
                "Agent %s: output validation failed after %d retries, missing fields: %s",
                self.name, max_retries, missing,
            )

        # 工具客观结果优先：若本轮有 run_python 失败，强制 vote_complete=false，防止 LLM 幻觉“全部通过”。
        if output:
            output = self._enforce_tool_result_honesty(output)

        elapsed = time.time() - t_start

        context.track_usage(
            prompt_tokens=last_usage.prompt_tokens if last_usage else 0,
            completion_tokens=last_usage.completion_tokens if last_usage else 0,
        )

        logger.info(
            "Agent %s: done in %.1fs, tokens=%dP/%dC, output_len=%d",
            self.name, elapsed,
            last_usage.prompt_tokens if last_usage else 0,
            last_usage.completion_tokens if last_usage else 0,
            len(last_response),
        )

        output.elapsed_time = elapsed

        # 记录本轮提示词详情到 context，供 trace 日志持久化
        if hasattr(context, "record_agent_prompt"):
            model = getattr(self.llm_adapter, "model", "")
            context.record_agent_prompt(
                agent_name=self.name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=messages,
                prompt_tokens=last_usage.prompt_tokens if last_usage else 0,
                completion_tokens=last_usage.completion_tokens if last_usage else 0,
                elapsed_time=elapsed,
                model=model,
            )

        # 注意：public_content 不在此处写入记忆，由 Graph._update_memories 统一处理
        # （与论文 Phase 4 一致：H_i^(t+1) ← H_i^(t) ⊕ m_pub,i ⊕ Σ m_priv,j）

        return output

    async def async_execute(
        self, input_data: Any, context: ExecutionContext, **kwargs
    ) -> AgentOutput:
        t_start = time.time()
        max_retries = kwargs.get("max_retries", 2)
        system_prompt, user_prompt = self._build_prompt(input_data, context)

        messages = self._chat_messages(system_prompt, user_prompt)

        if self.llm_adapter is None:
            logger.warning("Agent %s: no LLM adapter configured", self.name)
            return AgentOutput(
                public_content="[No LLM adapter configured]",
                raw_response="",
            )

        output = None
        last_response = ""
        last_usage = None
        executed_calls: list[dict] = []
        tool_schemas = self._build_tool_schemas() if self.tools else None

        for attempt in range(max_retries + 1):
            executed_calls = []  # 每轮重试隔离工具调用记录
            # 工具调用子循环：LLM 可通过原生 tool_calls 多次调用工具
            for tool_round in range(self.max_tool_rounds or MAX_TOOL_ROUNDS):
                logger.info(
                    "Agent %s (%s): calling LLM (attempt %d/%d, tool_round %d)...",
                    self.name, self.role, attempt + 1, max_retries + 1, tool_round + 1,
                )
                llm_response = await self.llm_adapter.agen(
                    messages,
                    tools=tool_schemas,
                )
                last_response = llm_response.content or ""
                last_usage = llm_response.usage

                if llm_response.tool_calls:
                    tool_summary = await self._process_native_tool_calls_async(llm_response.tool_calls, executed_calls)
                    # 重置 messages：智能体只看当前一次工具调用结果，不累积 Full Message History
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self._build_tool_followup_content(
                            context, user_prompt, tool_summary, final=False,
                        )},
                    ]
                    continue

                # 无 tool_calls：如果之前调用过工具，需要一次 JSON mode 收尾
                if executed_calls:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self._build_tool_followup_content(
                            context, user_prompt, self._build_tool_summary(executed_calls),
                            final=True,
                        )},
                    ]
                    logger.info(
                        "Agent %s: making final JSON-mode call after %d tool call(s)",
                        self.name, len(executed_calls),
                    )
                    llm_response = await self.llm_adapter.agen(
                        messages,
                        json_mode=True,
                    )
                    last_response = llm_response.content or ""
                    last_usage = llm_response.usage

                # 解析输出
                output = self._parse_response(last_response)
                output.tool_calls = executed_calls
                break
            else:
                # tool 循环耗尽
                if executed_calls:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": self._build_tool_followup_content(
                            context, user_prompt, self._build_tool_summary(executed_calls),
                            final=True,
                        )},
                    ]
                    llm_response = await self.llm_adapter.agen(
                        messages,
                        json_mode=True,
                    )
                    last_response = llm_response.content or ""
                    last_usage = llm_response.usage

                if last_response:
                    output = self._parse_response(last_response)
                    output.tool_calls = executed_calls
                else:
                    output = AgentOutput(
                        public_content="[Tool loop exhausted without final output]",
                        raw_response="",
                        tool_calls=executed_calls,
                    )

            missing = self._validate_output(output)
            if not missing:
                break

            if attempt < max_retries:
                logger.warning(
                    "Agent %s: missing fields %s, retrying...",
                    self.name, missing,
                )
                if not last_response or not last_response.strip():
                    feedback = (
                        "Your previous response was empty. "
                        "Please output a valid JSON object with ALL required fields. "
                        "Output ONLY the JSON, no markdown, no explanations."
                    )
                else:
                    missing_fields_str = ", ".join(missing)
                    feedback = (
                        f"Your previous response was invalid: "
                        f"missing required fields [{missing_fields_str}]. "
                        f"Please output a valid JSON object with ALL required fields. "
                        f"Output ONLY the JSON, no markdown, no explanations."
                    )
                messages.append({"role": "user", "content": feedback})

        if missing:
            logger.error(
                "Agent %s: output validation failed after %d retries, missing fields: %s",
                self.name, max_retries, missing,
            )

        # 工具客观结果优先：若本轮有 run_python 失败，强制 vote_complete=false，防止 LLM 幻觉“全部通过”。
        if output:
            output = self._enforce_tool_result_honesty(output)

        elapsed = time.time() - t_start

        context.track_usage(
            prompt_tokens=last_usage.prompt_tokens if last_usage else 0,
            completion_tokens=last_usage.completion_tokens if last_usage else 0,
        )

        logger.info(
            "Agent %s: done in %.1fs, tokens=%dP/%dC, output_len=%d",
            self.name, elapsed,
            last_usage.prompt_tokens if last_usage else 0,
            last_usage.completion_tokens if last_usage else 0,
            len(last_response),
        )

        output.elapsed_time = elapsed

        # 记录本轮提示词详情到 context，供 trace 日志持久化
        if hasattr(context, "record_agent_prompt"):
            model = getattr(self.llm_adapter, "model", "")
            context.record_agent_prompt(
                agent_name=self.name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                messages=messages,
                prompt_tokens=last_usage.prompt_tokens if last_usage else 0,
                completion_tokens=last_usage.completion_tokens if last_usage else 0,
                elapsed_time=elapsed,
                model=model,
            )

        # 注意：public_content 不在此处写入记忆，由 Graph._update_memories 统一处理

        return output
