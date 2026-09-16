"""消息路由策略模块。

提供多种路由策略：
- GraphRouting: 图邻接路由（沿边传递 public_content）
- SemanticRouting: 语义路由（优先 private_content，支持截断）
- BroadcastRouting: 广播路由（全对全通信）
- FreeTalkRouting: FT 显式 to 路由（方案一，不依赖语义匹配）
- IllocutionMatcherRouting: PragmaMAS 语旨匹配路由（通用）
- IllocutionMatcherMathRouting: 数学推理专用语旨匹配路由
- IllocutionMatcherCodeRouting: 代码生成专用语旨匹配路由
"""

from mas_topo.routing.base import RoutingStrategy
from mas_topo.routing.graph import GraphRouting
from mas_topo.routing.semantic import SemanticRouting
from mas_topo.routing.broadcast import BroadcastRouting
from mas_topo.routing.self_routing import SelfRouting
from mas_topo.routing.tlsg_routing import TlsgRouting

__all__ = [
    "RoutingStrategy",
    "GraphRouting",
    "SemanticRouting",
    "BroadcastRouting",
    "SelfRouting",
    "TlsgRouting",
]
