"""Ledger Memory Update — PragmaMAS-P2P 共享账本记忆注入策略。

核心逻辑：
- Agent 记忆 = 自身 public_content + 收到的 private/broadcast 消息。
- Ledger 摘要不存入记忆，而是注入 context.metadata 供 prompt 动态使用（每轮刷新当前状态视图，不累积历史）。
- 共享 ledger 由 SelfRouting 写入，本策略只负责按需查询。
"""

import json
from typing import Optional

from mas_topo._registries import memory_registry
from mas_topo.context import ExecutionContext
from mas_topo.ledger import IllocutionaryLedger
from mas_topo.memory.base import MemoryUpdateStrategy
from mas_topo.message import AgentOutput, Channel, Message

# context.metadata 中存储 per-agent ledger 摘要的键
LEDGER_SUMMARIES_KEY = "__ledger_summaries__"


@memory_registry.register(
    "ledger",
    display_name="Ledger 记忆更新",
    description="追加自身输出与收到的消息；ledger 摘要注入 context.metadata 供 prompt 动态使用。",
)
class LedgerMemoryUpdate(MemoryUpdateStrategy):
    """Inject messages into agent memory; inject ledger summaries into context metadata.

    Args:
        query_policy: ledger 摘要查询策略。
            - full: 注入完整 ledger（token 成本高）。
            - relevant_plus_pending: 注入与自己相关的条目 + 全局未闭环条目（默认）。
            - minimal: 仅注入与自己直接相关的条目。
        max_rounds_lookback: 摘要中最多保留最近多少轮的条目（默认 2）。
            设为 None 或 0 表示不限制。
        max_entries: 摘要中最多保留多少条 ledger 条目（默认 20）。
        content_snippet_length: 单条 ledger content 摘要长度（默认 80）。
    """

    _ACCEPTED_CHANNELS = frozenset({
        Channel.BROADCAST,
        Channel.FT_PRIVATE,
        "answer",  # SOP 等基线框架中显式传递的 answer 通道
    })

    def __init__(
        self,
        query_policy: str = "relevant_plus_pending",
        max_rounds_lookback: Optional[int] = 2,
        max_entries: int = 20,
        content_snippet_length: int = 80,
        inject_ledger_summary: str = "abnormal_only",
        tool_memory_mode: str = "result_only",
        **kwargs,
    ):
        _ = kwargs  # silently accept legacy params (max_parent_generations, memory_round_window, etc.)
        self.query_policy = query_policy.lower()
        self.max_rounds_lookback = max_rounds_lookback
        self.max_entries = max(max_entries, 1)
        self.content_snippet_length = max(content_snippet_length, 20)

        inject_ledger_summary = inject_ledger_summary.lower()
        if inject_ledger_summary not in ("always", "abnormal_only", "never"):
            raise ValueError(
                f"inject_ledger_summary must be always/abnormal_only/never, got {inject_ledger_summary}"
            )
        self.inject_ledger_summary = inject_ledger_summary

        tool_memory_mode = tool_memory_mode.lower()
        if tool_memory_mode not in ("verbose", "result_only", "summary", "none"):
            raise ValueError(
                f"tool_memory_mode must be verbose/result_only/summary/none, got {tool_memory_mode}"
            )
        self.tool_memory_mode = tool_memory_mode

    def update(
        self,
        agents: list,
        agent_names: list[str],
        routed_messages: dict[str, list[Message]],
        agent_outputs: Optional[dict[str, AgentOutput]] = None,
        round_num: int = 0,
        aggregation_order: Optional[list[int]] = None,
        context: Optional[ExecutionContext] = None,
    ) -> None:
        ledger = getattr(context, "ledger", None) if context else None
        tlsg_graph = getattr(context, "tlsg_graph", None) if context else None

        # 从 TLSG 图或 legacy ledger 提取最新全局 answer 快照，注入 context.metadata
        current_answer = ""
        current_answer_status = ""
        current_answer_source = ""
        if tlsg_graph is not None:
            current_answer, current_answer_status, current_answer_source = self._extract_answer_from_tlsg(tlsg_graph)
        elif ledger:
            current_answer, current_answer_status, current_answer_source = self._extract_latest_answer_with_status(ledger)
        if current_answer and context is not None:
            if not hasattr(context, "metadata") or context.metadata is None:
                context.metadata = {}
            context.metadata["current_answer"] = current_answer
            context.metadata["current_answer_status"] = current_answer_status
            # answer 来源 Agent 名，供 prompt 注入标签（answer_from_<name>）使用
            context.metadata["current_answer_source"] = current_answer_source

        # 构建 per-agent ledger 摘要，注入 context.metadata（不存入记忆）
        ledger_summaries = {}
        if ledger:
            for agent in agents:
                summary = self._build_summary(agent.name, agent_names, ledger)
                if summary:
                    ledger_summaries[agent.name] = summary
        if ledger_summaries and context is not None:
            if not hasattr(context, "metadata") or context.metadata is None:
                context.metadata = {}
            context.metadata[LEDGER_SUMMARIES_KEY] = ledger_summaries

        for agent in agents:
            private_msgs = [
                msg for msg in routed_messages.get(agent.name, [])
                if msg.channel in self._ACCEPTED_CHANNELS
            ]
            if aggregation_order and private_msgs:
                private_msgs = self._sort_messages_by_aggregation_order(
                    private_msgs, agent_names, aggregation_order
                )

            # Own outputs: 记录发出的 private_routing，使 Agent 追踪自己的通信历史
            if agent_outputs and agent.name in agent_outputs:
                output = agent_outputs[agent.name]
                routing_items = output.private_routing or []
                for item in routing_items:
                    if not isinstance(item, dict):
                        continue
                    recipients = item.get("recipients", [])
                    if isinstance(recipients, str):
                        recipients = [recipients]
                    illocution = str(item.get("illocution", "Assertive")).strip()
                    content = str(item.get("content", "")).strip()
                    if not content:
                        continue
                    recipient_str = ", ".join(recipients) if recipients else "?"
                    agent.append_memory(
                        f"[ROUND {round_num} - I sent to {recipient_str}]: "
                        f"[{illocution}]: {content}",
                        round_num=round_num,
                    )

                # 记录本 Agent 调用的工具及其结果，供后续轮次参考
                for entry in self._build_tool_memory_entries(output, round_num):
                    agent.append_memory(entry, round_num=round_num)

            # Routed messages
            for msg in private_msgs:
                label = "broadcast" if msg.channel == Channel.BROADCAST else "private"
                agent.append_memory(
                    f"[From {msg.sender_id} ({label})]: {msg.content}",
                    round_num=round_num,
                )

    @staticmethod
    def _extract_latest_answer(ledger: IllocutionaryLedger) -> Optional[str]:
        """从 ledger 所有条目中提取最新的非空 answer_snapshot。"""
        latest_answer = None
        latest_entry_id = -1
        for entry in ledger.all_entries():
            if entry.answer_snapshot and entry.entry_id > latest_entry_id:
                latest_answer = entry.answer_snapshot
                latest_entry_id = entry.entry_id
        return latest_answer

    @staticmethod
    def _extract_latest_answer_with_status(ledger: IllocutionaryLedger) -> tuple[str, str, str]:
        """从 ledger 所有条目中提取最新的非空 answer_snapshot 及其验证状态。

        返回 (answer, status, source)。status 取最新 entry 的 verification_status；
        如果后续有相同 answer_snapshot 的验证记录，会覆盖状态。
        source 为写入该 answer 快照的 entry 发送者。
        """
        latest_answer = ""
        latest_entry_id = -1
        answer_status = ""
        answer_source = ""
        status_by_answer: dict[str, str] = {}
        for entry in ledger.all_entries():
            if entry.answer_snapshot:
                if entry.verification_status:
                    status_by_answer[entry.answer_snapshot] = entry.verification_status
                if entry.entry_id > latest_entry_id:
                    latest_answer = entry.answer_snapshot
                    latest_entry_id = entry.entry_id
                    answer_source = entry.sender
        if latest_answer:
            answer_status = status_by_answer.get(latest_answer, "")
        return latest_answer, answer_status, answer_source

    @staticmethod
    def _extract_answer_from_tlsg(graph) -> tuple[str, str, str]:
        """从 TLSG 图所有节点中提取最新的 answer_snapshot 及其验证状态。

        返回 (answer, status, source)；source 为写入该快照的节点发送者。
        """
        best_answer = ""
        best_status = ""
        best_source = ""
        best_entry_id = -1
        for node in graph.all_nodes():
            if node.answer_snapshot and node.entry_id > best_entry_id:
                best_answer = node.answer_snapshot
                best_status = getattr(node, "verification_status", "") or ""
                best_source = node.sender
                best_entry_id = node.entry_id
        return best_answer, best_status, best_source

    def _build_tool_memory_entries(
        self, output: AgentOutput, round_num: int
    ) -> list[str]:
        """Build memory entries for tool calls made by *output*'s agent."""
        entries: list[str] = []
        if self.tool_memory_mode == "none":
            return entries

        tool_calls = getattr(output, "tool_calls", None) or []
        for call_info in tool_calls:
            if not isinstance(call_info, dict):
                continue
            call = call_info.get("call", {}) or {}
            tool_result = call_info.get("result", {}) or {}
            tool_name = call.get("name", "unknown")
            args = call.get("arguments", {})
            result_data = tool_result.get("result") if isinstance(tool_result, dict) else tool_result

            if isinstance(result_data, dict):
                passed = result_data.get("passed")
                stderr = result_data.get("stderr", "")
            else:
                passed = None
                stderr = ""

            if self.tool_memory_mode == "result_only":
                entry = f"  tool {tool_name}: passed={passed}"
                if passed is False and stderr:
                    entry += f", stderr={stderr[:100]}"
            elif self.tool_memory_mode == "summary":
                # 截断后的输入参数 + 结果：让 Agent 跨轮知道“跑了什么、结果如何”，
                # 同时避免 verbose 模式把完整代码灌进记忆。
                try:
                    args_text = json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    args_text = str(args)
                if len(args_text) > 200:
                    args_text = args_text[:200] + "..."
                entry = f"  tool {tool_name}: args={args_text} -> passed={passed}"
                if passed is False and stderr:
                    entry += f", stderr={stderr[:100]}"
            else:  # verbose
                if isinstance(result_data, dict):
                    summary = f"passed={passed} stderr={stderr[:200]}"
                else:
                    summary = str(result_data)[:200]
                entry = (
                    f"  tool {tool_name}: "
                    f"args={args} -> {summary}"
                )
            entries.append(entry)
        return entries

    def _build_summary(
        self, agent_name: str, agent_names: list[str], ledger: IllocutionaryLedger
    ) -> str:
        lines = [f"--- Global Contract Ledger Summary (relevant to {agent_name}) ---"]

        if self.inject_ledger_summary == "never":
            return ""

        if self.inject_ledger_summary == "abnormal_only":
            abnormal_items = ledger.query(status=["OVERDUE", "DISPUTED"])
            if not abnormal_items:
                return ""

        if self.query_policy == "full":
            relevant = ledger.all_entries()
        elif self.query_policy == "minimal":
            relevant = ledger.query(recipient=agent_name)
            relevant += [e for e in ledger.query(sender=agent_name) if e not in relevant]
        else:  # relevant_plus_pending (default)
            # Items directed at or sent by this agent
            relevant = ledger.query(recipient=agent_name)
            relevant += [e for e in ledger.query(sender=agent_name) if e not in relevant]

            # OVERDUE / DISPUTED 全量展示（异常状态需全员感知）；OPEN 已在 relevant 中覆盖
            status_items = ledger.query(status=["OVERDUE", "DISPUTED"])
            for item in status_items:
                if item not in relevant:
                    relevant.append(item)

        if not relevant:
            return ""

        # 按轮次回退截断：默认只保留最近 N 轮条目，避免早期已完成的指令无限堆积
        if self.max_rounds_lookback:
            current_round = max((e.round_num for e in relevant), default=0)
            min_round = max(1, current_round - self.max_rounds_lookback + 1)
            relevant = [e for e in relevant if e.round_num >= min_round]

        # 按 entry_id 排序后取最近 max_entries 条，进一步控制摘要长度
        relevant = sorted(relevant, key=lambda e: e.entry_id)[-self.max_entries :]

        snippet_len = self.content_snippet_length
        for entry in relevant:
            content = entry.content[:snippet_len]
            if len(entry.content) > snippet_len:
                content += "..."
            lines.append(
                f"[{entry.status}] {entry.entry_id}: {entry.sender} -> {', '.join(entry.recipients)} "
                f"[{entry.illocution}]: {content}"
            )
        return "\n".join(lines)

    def _sort_messages_by_aggregation_order(
        self,
        msgs: list[Message],
        agent_names: list[str],
        aggregation_order: list[int],
    ) -> list[Message]:
        name_to_idx = {name: i for i, name in enumerate(agent_names)}
        order_map = {idx: pos for pos, idx in enumerate(aggregation_order)}

        def key(msg: Message) -> int:
            idx = name_to_idx.get(msg.sender_id, len(agent_names))
            return order_map.get(idx, len(aggregation_order))

        return sorted(msgs, key=key)
