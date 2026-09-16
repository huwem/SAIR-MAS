"""消息路由策略抽象基类。"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np

from mas_topo.message import Message, AgentOutput
from mas_topo.context import ExecutionContext


class RoutingStrategy(ABC):
    """消息路由策略抽象基类。"""

    supports_selective_activation: bool = False

    def _state_backend(self, context: ExecutionContext) -> Any:
        """返回本路由策略用于记录语旨行为的状态对象。

        默认使用 ``context.ledger``（IllocutionaryLedger）。
        TLSG 路由会重写此方法以使用 ``context.tlsg_graph``，
        从而把 TLSG 图与旧版线性 ledger 在变量层面分离。
        """
        return getattr(context, "ledger", None)

    @abstractmethod
    def route(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
        aggregation_order: Optional[list[int]] = None,
    ) -> dict[str, list[Message]]:
        """根据拓扑和 Agent 输出，计算每个 Agent 应接收的消息。

        Args:
            agent_names: Agent 名称列表，顺序与 adjacency_matrix 索引一致。
            agent_outputs: 当前轮各 Agent 的输出。
            adjacency_matrix: 邻接矩阵 (N, N)，adj[i][j]=1 表示 i→j。
            context: 运行时上下文。
            aggregation_order: 论文 Algorithm 1 Phase 3 的聚合顺序 σ(t)，
                用于确定性地排列消息。

        Returns:
            {agent_name: [Message, ...]}
        """

    def get_active_workers(
        self,
        context: ExecutionContext,
        worker_names: list[str],
        previous_outputs: Optional[dict[str, AgentOutput]] = None,
        **kwargs,
    ) -> set[str]:
        """决定本轮应激活哪些 Worker。

        默认实现返回全部 Worker。子类可重写以实现选择性激活。

        Args:
            context: 运行时上下文（含 task, goal, round_num 等）。
            worker_names: 所有 Worker 名称列表。
            previous_outputs: 上一轮各 Worker 的输出。
            **kwargs: 扩展参数，由具体子类定义。

        Returns:
            应激活的 Worker 名称集合。
        """
        return set(worker_names)
