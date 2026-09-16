"""固定拓扑策略：full_connected, chain, star, ring, tree, mesh, layered, random。"""

import math
import random
from typing import Optional

import numpy as np

from mas_topo._registries import topology_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.topology.base import TopologyStrategy


class FixedTopology(TopologyStrategy):
    """预定义固定拓扑，支持多种经典网络结构。"""

    PATTERNS = [
        "full_connected", "chain", "star", "ring", "tree",
        "mesh", "layered", "random", "direct_answer",
    ]

    def __init__(self, pattern: str = "full_connected", seed: Optional[int] = None, manager_name: Optional[str] = None):
        if pattern not in self.PATTERNS:
            raise ValueError(
                f"Unknown pattern '{pattern}'. Available: {self.PATTERNS}"
            )
        self.pattern = pattern
        self.seed = seed
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

        if self.seed is not None:
            random.seed(self.seed)

        adj = self._build_pattern(self.pattern, N)
        adj = adj.astype(int)

        # star 模式支持以 manager_name 为中心节点（backward compatible）
        if self.pattern == "star" and self.manager_name and self.manager_name in agent_names:
            center_idx = agent_names.index(self.manager_name)
            if center_idx != 0:
                new_adj = np.zeros((N, N), dtype=int)
                for i in range(N):
                    if i != center_idx:
                        new_adj[center_idx][i] = 1
                        new_adj[i][center_idx] = 1
                adj = new_adj

        return adj, None

    @classmethod
    def _build_pattern(cls, pattern: str, N: int) -> np.ndarray:
        if pattern == "direct_answer":
            return np.zeros((1, 1), dtype=int)

        if pattern == "full_connected":
            adj = np.ones((N, N), dtype=int)
            np.fill_diagonal(adj, 0)
            return adj

        if pattern == "chain":
            adj = np.zeros((N, N), dtype=int)
            for i in range(N - 1):
                adj[i][i + 1] = 1
                adj[i + 1][i] = 1
            return adj

        if pattern == "star":
            adj = np.zeros((N, N), dtype=int)
            for i in range(1, N):
                adj[0][i] = 1
                adj[i][0] = 1
            return adj

        if pattern == "ring":
            adj = np.zeros((N, N), dtype=int)
            for i in range(N):
                adj[i][(i + 1) % N] = 1
            return adj

        if pattern == "tree":
            adj = np.zeros((N, N), dtype=int)
            for i in range(1, N):
                parent = (i - 1) // 2  # 二叉树
                adj[parent][i] = 1
                adj[i][parent] = 1
            return adj

        if pattern == "mesh":
            sqrt_n = int(math.sqrt(N))
            if sqrt_n * sqrt_n == N and N > 4:
                adj = np.zeros((N, N), dtype=int)
                for i in range(N):
                    if (i + 1) % sqrt_n != 0:
                        adj[i][i + 1] = adj[i + 1][i] = 1
                    if i < N - sqrt_n:
                        adj[i][i + sqrt_n] = adj[i + sqrt_n][i] = 1
                return adj
            # 非完美平方数退化为全连接
            adj = np.ones((N, N), dtype=int)
            np.fill_diagonal(adj, 0)
            return adj

        if pattern == "layered":
            adj = np.zeros((N, N), dtype=int)
            layer_num = max(2, N // 3)
            base = N // layer_num
            rem = N % layer_num
            layers = []
            for i in range(layer_num):
                size = base + (1 if i < rem else 0)
                layers.extend([i] * size)
            random.shuffle(layers)
            for i in range(N):
                for j in range(N):
                    if layers[j] == layers[i] + 1:
                        adj[i][j] = 1
            return adj

        if pattern == "random":
            adj = np.zeros((N, N), dtype=int)
            for i in range(N):
                for j in range(N):
                    if i != j:
                        adj[i][j] = random.randint(0, 1)
            return adj

        raise ValueError(f"Unknown pattern: {pattern}")


@topology_registry.register(
    "fixed",
    display_name="固定拓扑",
    description="预定义固定拓扑：full_connected, chain, star, ring, tree, mesh, layered, random",
    config_schema={
        "pattern": {"type": "str", "default": "full_connected", "description": "拓扑模式"},
        "seed": {"type": "int|null", "default": None, "description": "随机种子（random 模式用）"},
        "manager_name": {"type": "str|null", "default": None, "description": "star 模式下的中心节点名称"},
    },
)
class FixedTopologyRegistered(FixedTopology):
    """注册表入口。"""
    def __init__(self, pattern: str = "full_connected", seed: Optional[int] = None, manager_name: Optional[str] = None, **kwargs):
        super().__init__(pattern=pattern, seed=seed, manager_name=manager_name)
