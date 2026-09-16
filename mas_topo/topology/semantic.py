"""语义匹配拓扑策略（DyTopo 论文方法）。

使用 SentenceTransformer 嵌入 agent 的 query/key descriptor，
计算余弦相似度矩阵，按阈值和 max_in_degree 约束生成邻接矩阵。

当 manager_name 提供时，仅对 Worker 子集计算语义相似度，
邻接矩阵中 Manager 行/列置零（public 信道为固定星型，不走拓扑）。
"""

from typing import Optional
import threading

import numpy as np

from mas_topo._registries import topology_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.topology.base import TopologyStrategy

# 进程级缓存：避免多线程并发加载 + 每个实例重复加载
_engine_cache: dict[str, "SentenceTransformer"] = {}
_engine_lock = threading.Lock()


def _load_engine(model_path: str, cache_folder: str | None, local_only: bool):
    """线程安全地加载或获取缓存的 SentenceTransformer 实例。"""
    key = f"{model_path}|{cache_folder}|{local_only}"
    with _engine_lock:
        if key not in _engine_cache:
            import logging
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
            from sentence_transformers import SentenceTransformer
            _engine_cache[key] = SentenceTransformer(
                model_path,
                cache_folder=cache_folder,
                local_files_only=local_only,
            )
        return _engine_cache[key]


class SemanticMatchingTopology(TopologyStrategy):
    """基于语义匹配的动态拓扑构建。

    使用 query/key descriptor 嵌入计算 agent 间的语义相关性，
    阈值化和入度约束后生成有向通信图。

    当 manager_name 提供时，拓扑仅决定 Worker 间的 private 信道连接，
    Manager 从语义匹配中排除（Manager 行/列置零）。
    """

    def __init__(
        self,
        threshold: float = 0.3,
        max_in_degree: int = 3,
        embedding_model: str = "all-MiniLM-L6-v2",
        manager_name: Optional[str] = None,
    ):
        self.threshold = threshold
        self.max_in_degree = max_in_degree
        self.embedding_model = embedding_model
        self.manager_name = manager_name
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            import os
            model_path = os.environ.get("ST_MODEL_PATH", self.embedding_model)
            cache_folder = os.environ.get("ST_CACHE_FOLDER", None)
            local_only = os.environ.get("ST_LOCAL_ONLY", "").lower() in ("1", "true", "yes")
            self._engine = _load_engine(model_path, cache_folder, local_only)
        return self._engine

    def construct_topology(
        self,
        agent_names: list[str],
        agent_outputs: Optional[dict[str, AgentOutput]],
        context: ExecutionContext,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        N = len(agent_names)
        if N == 0:
            return np.zeros((0, 0), dtype=int), None

        # 确定 Worker 子集索引
        if self.manager_name and self.manager_name in agent_names:
            mgr_idx = agent_names.index(self.manager_name)
            worker_indices = [i for i in range(N) if i != mgr_idx]
        else:
            mgr_idx = -1
            worker_indices = list(range(N))

        W = len(worker_indices)

        # 首轮无输出时退化为全连接（仅 Worker 间）
        if agent_outputs is None:
            adj = np.ones((N, N), dtype=int)
            np.fill_diagonal(adj, 0)
            if mgr_idx >= 0:
                adj[mgr_idx, :] = 0
                adj[:, mgr_idx] = 0
            return adj, np.ones((N, N)) * 0.5

        # Worker 不足 2 个时无法构建 Worker 间拓扑
        if W <= 1:
            return np.zeros((N, N), dtype=int), np.zeros((N, N))

        # 仅对 Workers 收集描述符
        worker_names = [agent_names[i] for i in worker_indices]
        query_descs = []
        key_descs = []
        has_any_descriptor = False
        for name in worker_names:
            out = agent_outputs.get(name)
            if out and out.has_semantic_descriptors:
                query_descs.append(out.query_descriptor)
                key_descs.append(out.key_descriptor)
                has_any_descriptor = True
            elif out and out.public_content:
                desc = out.public_content[:200]
                query_descs.append(desc)
                key_descs.append(desc)
                has_any_descriptor = True
            else:
                query_descs.append("")
                key_descs.append("")

        # 所有 Worker 都没有可用的语义描述符时，退化为全连接（仅 Worker 间）
        if not has_any_descriptor:
            adj = np.zeros((N, N), dtype=int)
            rel = np.zeros((N, N))
            for i in worker_indices:
                for j in worker_indices:
                    if i != j:
                        adj[i][j] = 1
                        rel[i][j] = 0.5
            return adj, rel

        # 计算 W x W 相似度矩阵
        engine = self._get_engine()
        query_embeds = engine.encode(query_descs, convert_to_numpy=True, show_progress_bar=False)
        key_embeds = engine.encode(key_descs, convert_to_numpy=True, show_progress_bar=False)

        q_norms = query_embeds / (np.linalg.norm(query_embeds, axis=1, keepdims=True) + 1e-8)
        k_norms = key_embeds / (np.linalg.norm(key_embeds, axis=1, keepdims=True) + 1e-8)

        # R[i,j] = k_i · q_j: agent i 的 key 匹配 agent j 的 query
        # → agent i 拥有 agent j 需要的信息 → 边 i→j（i 发送给 j）
        worker_relevance = np.dot(k_norms, q_norms.T)

        worker_adj = (worker_relevance > self.threshold).astype(int)
        np.fill_diagonal(worker_adj, 0)

        if self.max_in_degree is not None:
            for jw in range(W):
                incoming_scores = worker_relevance[:, jw]
                incoming_indices = np.argsort(incoming_scores)[::-1]
                active = []
                for iw in incoming_indices:
                    if iw != jw and worker_adj[iw][jw] == 1:
                        active.append(iw)
                    if len(active) >= self.max_in_degree:
                        break
                worker_adj[:, jw] = 0
                for iw in active:
                    worker_adj[iw][jw] = 1

        # 嵌入 N x N 矩阵
        adj = np.zeros((N, N), dtype=int)
        rel = np.zeros((N, N))
        for wi, i in enumerate(worker_indices):
            for wj, j in enumerate(worker_indices):
                adj[i][j] = worker_adj[wi][wj]
                rel[i][j] = worker_relevance[wi][wj]

        return adj, rel

    def compute_aggregation_order(self, adjacency_matrix: np.ndarray) -> list[int]:
        """论文 Algorithm 1 Phase 3 + Section 3.4.2：计算聚合顺序 σ(t)。

        Case I (DAG): 拓扑排序，字典序打破平局。
        Case II (Cyclic): 贪心破环 — 每次选受限入度最小的节点。
        """
        N = len(adjacency_matrix)
        if N == 0:
            return []

        adj = adjacency_matrix.copy()

        # 检测是否有环
        if self._is_dag(adj):
            return self._topological_sort(adj)
        else:
            return self._greedy_cycle_break(adj)

    @staticmethod
    def _is_dag(adj: np.ndarray) -> bool:
        """Kahn 算法检测 DAG。"""
        N = len(adj)
        in_degree = adj.sum(axis=0)
        queue = [i for i in range(N) if in_degree[i] == 0]
        visited = 0
        while queue:
            u = queue.pop(0)
            visited += 1
            for v in range(N):
                if adj[u][v]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        queue.append(v)
        return visited == N

    @staticmethod
    def _topological_sort(adj: np.ndarray) -> list[int]:
        """Kahn 拓扑排序，字典序打破平局。"""
        N = len(adj)
        in_degree = adj.sum(axis=0).tolist()
        queue = sorted([i for i in range(N) if in_degree[i] == 0])
        order = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in range(N):
                if adj[u][v]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        # 按索引排序插入以保持确定性
                        inserted = False
                        for qi in range(len(queue)):
                            if v < queue[qi]:
                                queue.insert(qi, v)
                                inserted = True
                                break
                        if not inserted:
                            queue.append(v)
        return order

    @staticmethod
    def _greedy_cycle_break(adj: np.ndarray) -> list[int]:
        """论文 Equation (10)-(11)：贪心破环启发式。

        每次从剩余节点 U 中选择受限入度 d_in(i; U) 最小的节点，
        附加节点到 σ(t) 并从 U 中移除，直到所有节点放置完毕。
        """
        N = len(adj)
        U = set(range(N))
        order = []

        while U:
            # 计算每个剩余节点的受限入度
            best_i = -1
            best_degree = N + 1
            for i in sorted(U):  # sorted for deterministic tie-breaking
                restricted_in = sum(
                    1 for j in U if j != i and adj[j][i] == 1
                )
                if restricted_in < best_degree:
                    best_degree = restricted_in
                    best_i = i
            order.append(best_i)
            U.remove(best_i)

        return order


@topology_registry.register(
    "semantic",
    display_name="语义匹配拓扑",
    description="基于 SentenceTransformer 语义匹配的动态拓扑（DyTopo 论文方法）。"
                "设置 manager_name 后仅计算 Worker 间拓扑",
    config_schema={
        "threshold": {"type": "float", "default": 0.3, "description": "相似度阈值"},
        "max_in_degree": {"type": "int", "default": 3, "description": "最大入度约束"},
        "embedding_model": {"type": "str", "default": "all-MiniLM-L6-v2", "description": "嵌入模型名"},
        "manager_name": {"type": "str|null", "default": None, "description": "Manager 名称，排除 Manager 后仅计算 Worker 间拓扑"},
    },
)
class SemanticMatchingTopologyRegistered(SemanticMatchingTopology):
    def __init__(self, threshold: float = 0.3, max_in_degree: int = 3,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 manager_name: Optional[str] = None, **kwargs):
        super().__init__(threshold=threshold, max_in_degree=max_in_degree,
                         embedding_model=embedding_model, manager_name=manager_name)
