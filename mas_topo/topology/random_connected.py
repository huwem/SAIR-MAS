"""随机连通拓扑：每轮动态生成随机连通有向图。"""

import random
from typing import Optional

import numpy as np

from mas_topo._registries import topology_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.topology.base import TopologyStrategy


class RandomConnectedTopology(TopologyStrategy):
    """每轮生成随机连通有向图。

    连通性保证：先生成随机排列构成有向环，再以 extra_edge_prob 概率添加额外边。
    """

    def __init__(self, extra_edge_prob: float = 0.3, seed: Optional[int] = None):
        self.extra_edge_prob = extra_edge_prob
        self.seed = seed

    def construct_topology(
        self,
        agent_names: list[str],
        agent_outputs: Optional[dict[str, AgentOutput]],
        context: ExecutionContext,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        N = len(agent_names)
        if N <= 1:
            adj = np.zeros((N, N), dtype=int)
            return adj, None

        rng = random.Random(self.seed)
        # 每轮使用不同的种子：结合固定 seed + 当前轮次
        round_num = getattr(context, "round_num", 0) if context else 0
        if self.seed is not None:
            rng = random.Random(self.seed * 10000 + round_num)

        adj = np.zeros((N, N), dtype=int)

        # 随机排列构成有向环，保证强连通
        perm = list(range(N))
        rng.shuffle(perm)
        for k in range(N):
            src, dst = perm[k], perm[(k + 1) % N]
            adj[src][dst] = 1

        # 额外随机边
        for i in range(N):
            for j in range(N):
                if i != j and adj[i][j] == 0 and rng.random() < self.extra_edge_prob:
                    adj[i][j] = 1

        return adj, None


@topology_registry.register(
    "random_connected",
    display_name="随机连通拓扑",
    description="每轮动态生成随机连通有向图（随机环 + 额外边），保证无孤立节点",
    config_schema={
        "extra_edge_prob": {
            "type": "float",
            "default": 0.3,
            "description": "额外边添加概率",
        },
        "seed": {
            "type": "int|null",
            "default": None,
            "description": "随机种子",
        },
    },
)
class RandomConnectedTopologyRegistered(RandomConnectedTopology):
    """注册表入口。"""

    def __init__(
        self, extra_edge_prob: float = 0.3, seed: Optional[int] = None, **kwargs
    ):
        super().__init__(extra_edge_prob=extra_edge_prob, seed=seed)
