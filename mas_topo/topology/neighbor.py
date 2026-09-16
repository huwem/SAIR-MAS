"""邻居拓扑策略。

基于各 Agent 配置中的邻居表构建通信拓扑邻接矩阵。
"""

from typing import Optional

import numpy as np

from mas_topo._registries import topology_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.topology.base import TopologyStrategy


class NeighborTopology(TopologyStrategy):
    """基于邻居表的拓扑策略。

    从配置中读取每个 Agent 的邻居列表，构建二值邻接矩阵。
    adj[i][j] = 1 表示 agent_i 可以向 agent_j 发送私密消息。

    Args:
        agent_neighbor_map: {agent_name: [neighbor_name, ...]} 邻居映射。
            若未提供，则退化为全连接。
        manager_name: Manager Agent 名称。若提供，拓扑仅反映 Worker 间连接，
            Manager 与所有 Worker 之间默认双向连接。
    """

    strategy_name = "neighbor"

    def __init__(
        self,
        agent_neighbor_map: Optional[dict[str, list[str]]] = None,
        manager_name: Optional[str] = None,
    ):
        self.agent_neighbor_map = agent_neighbor_map or {}
        self.manager_name = manager_name

    def construct_topology(
        self,
        agent_names: list[str],
        agent_outputs: Optional[dict[str, AgentOutput]],
        context: ExecutionContext,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        N = len(agent_names)
        if N == 0:
            return np.zeros((0, 0), dtype=int), None

        adj = np.zeros((N, N), dtype=int)
        name_to_idx = {name: i for i, name in enumerate(agent_names)}

        # 优先使用显式邻居映射
        if self.agent_neighbor_map:
            for i, src_name in enumerate(agent_names):
                neighbors = self.agent_neighbor_map.get(src_name, [])
                for neighbor_name in neighbors:
                    j = name_to_idx.get(neighbor_name)
                    if j is not None and i != j:
                        adj[i][j] = 1
            return adj, None

        # 否则从上下文 Agent 配置的 neighbors 字段构建
        agents = getattr(context, "agents", []) or []
        agent_by_name = {a.name: a for a in agents if hasattr(a, "name")}

        for i, src_name in enumerate(agent_names):
            agent = agent_by_name.get(src_name)
            if not agent:
                continue
            for nb in getattr(agent, "neighbors", []) or []:
                nb_name = nb.get("name") if isinstance(nb, dict) else nb
                j = name_to_idx.get(nb_name)
                if j is not None and i != j:
                    adj[i][j] = 1

        # 若仍未得到任何边，退化为全连接（避免空矩阵导致无通信）
        if adj.sum() == 0:
            adj = np.ones((N, N), dtype=int)
            np.fill_diagonal(adj, 0)

        return adj, None


@topology_registry.register(
    "neighbor",
    display_name="邻居拓扑",
    description="基于各 Agent 配置中的邻居表构建通信拓扑邻接矩阵",
    config_schema={
        "agent_neighbor_map": {
            "type": "object|null",
            "default": None,
            "description": "Agent 邻居映射 {agent_name: [neighbor_name, ...]}",
        },
        "manager_name": {
            "type": "str|null",
            "default": None,
            "description": "Manager Agent 名称",
        },
    },
)
class NeighborTopologyRegistered(NeighborTopology):
    """注册表入口。"""

    def __init__(
        self,
        agent_neighbor_map: Optional[dict[str, list[str]]] = None,
        manager_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            agent_neighbor_map=agent_neighbor_map,
            manager_name=manager_name,
        )
