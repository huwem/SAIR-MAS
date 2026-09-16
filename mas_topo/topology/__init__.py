"""拓扑策略模块。

提供多种拓扑构建策略：
- FixedTopology: 固定拓扑 (full_connected, chain, star, ring, tree, mesh, layered, random)
- SemanticMatchingTopology: 语义匹配拓扑 (DyTopo 方法)
- CosineTopology: 余弦相似度拓扑 (PTopo 方法)
- ManualTopology: 手动指定邻接矩阵
- NeighborTopology: 基于邻居表的约束拓扑
"""

from mas_topo.topology.base import TopologyStrategy
from mas_topo.topology.fixed import FixedTopology
from mas_topo.topology.semantic import SemanticMatchingTopology
from mas_topo.topology.cosine import CosineTopology
from mas_topo.topology.manual import ManualTopology
from mas_topo.topology.neighbor import NeighborTopology
from mas_topo.topology.random_connected import RandomConnectedTopology

__all__ = [
    "TopologyStrategy",
    "FixedTopology",
    "SemanticMatchingTopology",
    "CosineTopology",
    "ManualTopology",
    "NeighborTopology",
    "RandomConnectedTopology",
]
