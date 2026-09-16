"""余弦相似度拓扑策略（PTopo 方法）。

计算 agent 角色描述嵌入与 query 嵌入的余弦相似度，
边 (i→j) 的权重 = (sim_i + sim_j) / 2。
"""

from typing import Optional

import numpy as np

from mas_topo._registries import topology_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.topology.base import TopologyStrategy
from mas_topo.topology.semantic import _load_engine


class CosineTopology(TopologyStrategy):
    """基于余弦相似度的动态拓扑构建。

    计算每个 agent 与 task 的相似度，边权重为两端 agent 相似度的平均。
    """

    def __init__(
        self,
        threshold: float = 0.3,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.threshold = threshold
        self.embedding_model = embedding_model
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

        if not context.task:
            adj = np.ones((N, N), dtype=int)
            np.fill_diagonal(adj, 0)
            return adj, np.ones((N, N)) * 0.5

        engine = self._get_engine()

        role_descs = []
        for name in agent_names:
            out = agent_outputs.get(name) if agent_outputs else None
            if out and out.key_descriptor:
                role_descs.append(out.key_descriptor)
            else:
                role_descs.append(name)
        agent_embeds = engine.encode(role_descs, convert_to_numpy=True, show_progress_bar=False)

        query_embed = engine.encode([context.task], convert_to_numpy=True, show_progress_bar=False)[0]

        cos_sim = np.array([
            np.dot(agent_embeds[i], query_embed)
            / (np.linalg.norm(agent_embeds[i]) * np.linalg.norm(query_embed) + 1e-8)
            for i in range(N)
        ])

        spatial_logits = (cos_sim[np.newaxis, :] + cos_sim[:, np.newaxis]) / 2

        min_val = spatial_logits.min()
        max_val = spatial_logits.max()
        if max_val - min_val > 1e-8:
            spatial_logits = (spatial_logits - min_val) / (max_val - min_val) * 2 - 1

        adj = (spatial_logits > self.threshold).astype(int)
        np.fill_diagonal(adj, 0)

        return adj, spatial_logits


@topology_registry.register(
    "cosine",
    display_name="余弦相似度拓扑",
    description="基于角色描述与任务余弦相似度的动态拓扑（PTopo 方法）",
    config_schema={
        "threshold": {"type": "float", "default": 0.3, "description": "相似度阈值"},
        "embedding_model": {"type": "str", "default": "all-MiniLM-L6-v2", "description": "嵌入模型名"},
    },
)
class CosineTopologyRegistered(CosineTopology):
    def __init__(self, threshold: float = 0.3,
                 embedding_model: str = "all-MiniLM-L6-v2", **kwargs):
        super().__init__(threshold=threshold, embedding_model=embedding_model)
