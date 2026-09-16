"""TLSG memory update — simple message injection (v3).

One injection per round per agent: received DAG nodes injected as simple
one-line entries (classic p2p_free style), each node at most once ever
(permanent dedup).

Follows the classic memory pattern: messages accumulate naturally as simple
lines.
"""

import logging
from typing import Optional

from mas_topo._registries import memory_registry
from mas_topo.context import ExecutionContext
from mas_topo.memory.ledger_memory import LedgerMemoryUpdate
from mas_topo.message import AgentOutput, Message
from mas_topo.tlsg.graph import IllocutionNode, TemporalLayeredSpeechActGraph

logger = logging.getLogger(__name__)


@memory_registry.register(
    "tlsg_memory",
    display_name="TLSG DAG 语旨记忆",
    description="经典消息注入风格：每个 Agent 收到的语旨节点作为简洁单行消息注入，终身去重。",
)
class TlsgMemoryUpdate(LedgerMemoryUpdate):
    """Speech-act-aware memory: simple message injection.

    - Message injection: received DAG nodes as simple one-liners
      (like classic p2p_free), permanent dedup across all rounds.
    """

    def __init__(
        self,
        query_policy: str = "relevant_plus_pending",
        max_rounds_lookback: Optional[int] = 2,
        max_entries: int = 20,
        content_snippet_length: int = 2000,
        inject_ledger_summary: str = "always",
        tool_memory_mode: str = "result_only",
        max_parent_generations: int = 3,
        memory_round_window: int = 3,
    ):
        super().__init__(
            query_policy=query_policy,
            max_rounds_lookback=max_rounds_lookback,
            max_entries=max_entries,
            content_snippet_length=content_snippet_length,
            inject_ledger_summary=inject_ledger_summary,
            tool_memory_mode=tool_memory_mode,
        )
        self.max_parent_generations = max(0, max_parent_generations)
        self.memory_round_window = max(1, memory_round_window)
        self._cached_answer: str = ""
        self._cached_answer_status: str = ""
        self._cached_answer_source: str = ""
        # Per-agent set of already-injected DAG nodes, persisted across rounds
        # so that the same node is never repeated in memory.
        self._traced_nodes: dict[str, set[int]] = {}

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
        graph = (
            getattr(context, "tlsg_graph", None)
            or getattr(context, "ledger", None)
            if context
            else None
        )
        is_tlsg = isinstance(graph, TemporalLayeredSpeechActGraph)

        # --- current_answer from TLSG graph → context.metadata ---
        if is_tlsg:
            self._extract_answer_from_tlsg(graph, context)

        # --- TLSG DAG memory: per-received-node chain ---
        if is_tlsg:
            self._inject_dag_memory(agents, graph, round_num)

    # -- answer extraction ---------------------------------------------------

    def _extract_answer_from_tlsg(
        self,
        graph: TemporalLayeredSpeechActGraph,
        context: Optional[ExecutionContext],
    ) -> None:
        best_node = None
        for node in graph.all_nodes():
            if node.answer_snapshot and (best_node is None or node.entry_id > best_node.entry_id):
                best_node = node
        if best_node is not None:
            self._cached_answer = best_node.answer_snapshot
            self._cached_answer_status = best_node.verification_status or ""
            self._cached_answer_source = best_node.sender

        if context is not None:
            if not hasattr(context, "metadata") or context.metadata is None:
                context.metadata = {}
            context.metadata["current_answer"] = self._cached_answer
            context.metadata["current_answer_status"] = self._cached_answer_status
            # answer 来源 Agent 名，供 prompt 注入标签（answer_from_<name>）使用
            context.metadata["current_answer_source"] = self._cached_answer_source

    # -- DAG message injection ------------------------------------------------

    def _inject_dag_memory(
        self,
        agents: list,
        graph: TemporalLayeredSpeechActGraph,
        round_num: int,
    ) -> None:
        """Inject each received DAG node as a simple one-line memory entry.

        Classic p2p_free style: one line per message, permanent dedup via
        _traced_nodes.  No chain building, no obligation grouping.
        Messages are injected in full — no content truncation.
        """
        for agent in agents:
            received = sorted(
                [
                    n for n in graph.all_nodes()
                    if agent.name in n.recipients and n.round <= round_num
                ],
                key=lambda n: n.entry_id,
            )
            traced = self._traced_nodes.setdefault(agent.name, set())
            for node in received:
                if node.entry_id in traced:
                    continue
                traced.add(node.entry_id)
                # 消息全文注入，不做内容截断（截断会导致代码/推导信息缺失）
                agent.append_memory(
                    f"[From {node.sender} (R{node.round})]: "
                    f"[{node.illocution}]: {node.content}",
                    round_num=round_num,
                )
