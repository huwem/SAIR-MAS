"""图邻接路由策略。

沿邻接矩阵的边传递 public_content。
"""

import numpy as np

from mas_topo._registries import routing_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import Message, AgentOutput
from mas_topo.routing.base import RoutingStrategy


class GraphRouting(RoutingStrategy):
    """基于图结构的路由。

    沿邻接矩阵的边传递 public_content。
    adj[i][j]=1 时，agent_i 的输出传给 agent_j。
    """

    def route(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
        aggregation_order=None,
    ) -> dict[str, list[Message]]:
        N = len(agent_names)
        result = {name: [] for name in agent_names}

        for i in range(N):
            for j in range(N):
                if adjacency_matrix[i][j] == 1:
                    src_name = agent_names[i]
                    dst_name = agent_names[j]
                    src_out = agent_outputs.get(src_name)
                    if src_out and src_out.public_content:
                        result[dst_name].append(Message(
                            content=src_out.public_content,
                            sender_id=src_name,
                            receiver_id=dst_name,
                            channel="spatial",
                        ))
        return result


@routing_registry.register(
    "graph",
    display_name="图路由",
    description="沿邻接矩阵边传递 public_content",
    config_schema={},
)
class GraphRoutingRegistered(GraphRouting):
    def __init__(self, **kwargs):
        super().__init__()
