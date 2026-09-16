"""语义路由策略（论文 Section 3.2 双信道设计）。

双信道路由：
- public 信道（semantic-public）：
  - Phase 1a: Worker → Manager 单向。Worker 的 public_content 汇总给 Manager，
    供 Phase 5 全局决策（论文 Line 28: S_global ← [C_task; Σ m_pub,i]）。
  - Phase 1b: Manager → Workers 广播本轮目标 context.goal（即 C_task），
    而非 Manager 的 public_content。Manager 的 public_content 只用于自身记忆更新。
- private 信道（semantic-private）：沿邻接矩阵的 Worker→Worker 边传递。
  仅传递 private_content，不做 public_content fallback（论文 Phase 4 精确实现）。

当 manager_name 未提供时，回退到旧语义（沿所有边统一路由）。
"""

from typing import Optional

import numpy as np

from mas_topo._registries import routing_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import Message, AgentOutput
from mas_topo.routing.base import RoutingStrategy


class SemanticRouting(RoutingStrategy):
    """基于语义描述符的双信道路由。

    当 manager_name 提供时启用双信道：
    - semantic-public: Worker → Manager 星型上行（Phase 1a），Manager → Workers
      广播本轮目标 context.goal（Phase 1b）。Manager 的 public_content 不参与广播。
    - semantic-private: 沿邻接矩阵 Worker→Worker 边传递 private_content，
      不做 public_content fallback（严格遵循论文 Phase 4）。

    当 manager_name 为 None 时回退到旧语义（沿所有边统一路由）。
    """

    def __init__(self, memory_truncate: int = 500, manager_name: Optional[str] = None):
        self.memory_truncate = memory_truncate
        self.manager_name = manager_name

    def route(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
        aggregation_order: Optional[list[int]] = None,
    ) -> dict[str, list[Message]]:
        N = len(agent_names)
        result = {name: [] for name in agent_names}

        # 双信道语义：manager_name 已提供且在 agent_names 中
        if self.manager_name and self.manager_name in agent_names:
            return self._route_dual_channel(
                agent_names, agent_outputs, adjacency_matrix,
                context, result, aggregation_order,
            )

        # 旧语义回退：manager_name 未提供
        return self._route_legacy(
            agent_names, agent_outputs, adjacency_matrix,
            context, result, aggregation_order,
        )

    def _route_dual_channel(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
        result: dict[str, list[Message]],
        aggregation_order: Optional[list[int]] = None,
    ) -> dict[str, list[Message]]:
        N = len(agent_names)
        trunc = self.memory_truncate

        # Phase 1a: public 信道 — Worker → Manager
        # 论文 Section 3.2: public messages visible to Manager for global monitoring
        for i in range(N):
            src = agent_names[i]
            if src == self.manager_name:
                continue
            src_out = agent_outputs.get(src)
            if src_out and src_out.public_content:
                result[self.manager_name].append(Message(
                    content=src_out.public_content[:trunc],
                    sender_id=src,
                    receiver_id=self.manager_name,
                    channel="semantic-public",
                    metadata={
                        "query_descriptor": src_out.query_descriptor,
                        "key_descriptor": src_out.key_descriptor,
                    },
                ))

        # Phase 1b: public 信道 — Manager → Workers 广播本轮目标
        # 论文设计：Manager 的 public_content 只用于自身记忆更新和 Phase 5 全局决策，
        # 不广播给 Worker。Worker 通过 context.goal（即 C_task）获得本轮目标。
        if context.goal:
            for i in range(N):
                dst = agent_names[i]
                if dst == self.manager_name:
                    continue
                result[dst].append(Message(
                    content=context.goal[:trunc],
                    sender_id=self.manager_name,
                    receiver_id=dst,
                    channel="semantic-public",
                    metadata={"round_goal": True},
                ))

        # Phase 2: private 信道 — 沿邻接矩阵 Worker→Worker 边
        # 论文 Phase 4 精确实现: H_i ← H_i ⊕ m_pub,i ⊕ Σ({m_priv,j | j ∈ N_in(i)})
        # 仅传递 private_content[dst]，不做 public_content fallback。
        # Worker 的 public_content 只用于自身记忆更新（_update_memories 中处理）
        # 和 Manager 全局视图（Phase 1a）。不沿邻接矩阵传播。
        # 附带 sender 的 answer 字段（如 Developer 的代码），确保接收方可见。
        for i in range(N):
            src = agent_names[i]
            if src == self.manager_name:
                continue
            src_out = agent_outputs.get(src)
            if not src_out:
                continue
            for j in range(N):
                dst = agent_names[j]
                if dst == self.manager_name:
                    continue
                if adjacency_matrix[i][j] == 1:
                    private_msg = src_out.private_content.get(dst, "")
                    if not private_msg:
                        continue
                    if src_out.answer and str(src_out.answer).strip():
                        private_msg = private_msg + f"\n\n[Answer]:\n{src_out.answer}"

                    result[dst].append(Message(
                        content=private_msg[:trunc],
                        sender_id=src,
                        receiver_id=dst,
                        channel="semantic-private",
                        metadata={
                            "query_descriptor": src_out.query_descriptor,
                            "key_descriptor": src_out.key_descriptor,
                        },
                    ))

        return result

    def _route_legacy(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
        result: dict[str, list[Message]],
        aggregation_order: Optional[list[int]] = None,
    ) -> dict[str, list[Message]]:
        N = len(agent_names)
        trunc = self.memory_truncate

        for i in range(N):
            for j in range(N):
                if adjacency_matrix[i][j] == 1:
                    src_name = agent_names[i]
                    dst_name = agent_names[j]
                    src_out = agent_outputs.get(src_name)
                    if not src_out:
                        continue

                    private_msg = src_out.private_content.get(dst_name, "")
                    if not private_msg:
                        continue
                    if src_out.answer and str(src_out.answer).strip():
                        private_msg = private_msg + f"\n\n[Answer]:\n{src_out.answer}"

                    result[dst_name].append(Message(
                        content=private_msg[:trunc],
                        sender_id=src_name,
                        receiver_id=dst_name,
                        channel="semantic",
                        metadata={
                            "query_descriptor": src_out.query_descriptor,
                            "key_descriptor": src_out.key_descriptor,
                        },
                    ))
        return result


@routing_registry.register(
    "semantic",
    display_name="语义路由",
    description="沿邻接矩阵边传递消息，优先 private_content，支持语义描述符。"
                "设置 manager_name 启用双信道（public: Worker↔Manager, private: Worker→Worker）",
    config_schema={
        "memory_truncate": {"type": "int", "default": 500, "description": "消息截断长度"},
        "manager_name": {"type": "str|null", "default": None, "description": "Manager 名称，启用双信道路由"},
    },
)
class SemanticRoutingRegistered(SemanticRouting):
    def __init__(self, memory_truncate: int = 500, manager_name: Optional[str] = None, **kwargs):
        super().__init__(memory_truncate=memory_truncate, manager_name=manager_name)
