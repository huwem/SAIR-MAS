"""手动邻接矩阵拓扑策略。

从配置直接指定邻接矩阵，用于精确控制拓扑结构。
"""

from typing import Optional

import numpy as np

from mas_topo._registries import topology_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.topology.base import TopologyStrategy


class ManualTopology(TopologyStrategy):
    """从配置直接指定邻接矩阵。

    用于精确控制拓扑结构（如从文件加载或手工指定）。
    """

    def __init__(self, adjacency: Optional[list[list[int]]] = None):
        self._adjacency = np.array(adjacency, dtype=int) if adjacency else None

    def construct_topology(
        self,
        agent_names: list[str],
        agent_outputs: Optional[dict[str, AgentOutput]],
        context: ExecutionContext,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        N = len(agent_names)
        if self._adjacency is not None:
            adj = self._adjacency
            if adj.shape != (N, N):
                raise ValueError(
                    f"Adjacency matrix shape {adj.shape} doesn't match "
                    f"agent count {N}"
                )
            return adj.copy(), None

        adj = np.ones((N, N), dtype=int)
        np.fill_diagonal(adj, 0)
        return adj, None


@topology_registry.register(
    "manual",
    display_name="手动拓扑",
    description="通过显式邻接矩阵精确控制通信拓扑",
    config_schema={
        "adjacency": {"type": "list[list[int]]|null", "default": None, "description": "邻接矩阵"},
    },
)
class ManualTopologyRegistered(ManualTopology):
    def __init__(self, adjacency: Optional[list[list[int]]] = None, **kwargs):
        super().__init__(adjacency=adjacency)
