"""拓扑构建策略抽象基类。"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from mas_topo.message import AgentOutput
from mas_topo.context import ExecutionContext


class TopologyStrategy(ABC):
    """拓扑构建策略抽象基类。

    每种策略实现 construct_topology 方法，返回邻接矩阵和权重矩阵。
    """

    @abstractmethod
    def construct_topology(
        self,
        agent_names: list[str],
        agent_outputs: Optional[dict[str, AgentOutput]],
        context: ExecutionContext,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """构建当前轮次的通信拓扑。

        Args:
            agent_names: Agent 名称列表，顺序决定矩阵索引。
            agent_outputs: 当前轮各 Agent 的输出（首轮可能为 None）。
            context: 运行时上下文。

        Returns:
            (adjacency_matrix, relevance_matrix)
            adjacency_matrix: 二值邻接矩阵 (N, N)，adj[i][j]=1 表示 i→j 有边。
            relevance_matrix: 权重矩阵 (N, N)，无权重时返回 None。
        """

    def compute_aggregation_order(self, adjacency_matrix: np.ndarray) -> list[int]:
        """根据邻接矩阵计算聚合顺序 σ(t)（论文 Algorithm 1 Phase 3）。

        默认实现按入度升序排列（相当于贪心破环的近似）。

        子类可覆写以实现拓扑排序或更精确的破环启发式。

        Args:
            adjacency_matrix: 邻接矩阵 (N, N)，adj[i][j]=1 表示 i→j。

        Returns:
            agent 索引列表，按聚合顺序排列。
        """
        N = len(adjacency_matrix)
        if N == 0:
            return []
        in_degrees = adjacency_matrix.sum(axis=0)
        return sorted(range(N), key=lambda i: in_degrees[i])
