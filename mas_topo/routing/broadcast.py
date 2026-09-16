"""广播路由策略。

忽略拓扑结构，将所有 agent 的 public_content 广播给所有其他 agent。
"""

import numpy as np

from mas_topo._registries import routing_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import Message, AgentOutput
from mas_topo.routing.base import RoutingStrategy


class BroadcastRouting(RoutingStrategy):
    """广播路由。

    忽略邻接矩阵，每个 agent 的 public_content 与 answer 广播给所有其他 agent。
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
            src_name = agent_names[i]
            src_out = agent_outputs.get(src_name)
            if not src_out:
                continue
            contents = []
            if src_out.public_content:
                contents.append((src_out.public_content, "broadcast"))
            if src_out.answer:
                contents.append((src_out.answer, "answer"))
            if not contents:
                continue
            for j in range(N):
                if i == j:
                    continue
                dst_name = agent_names[j]
                for content, channel in contents:
                    result[dst_name].append(Message(
                        content=content,
                        sender_id=src_name,
                        receiver_id=dst_name,
                        channel=channel,
                    ))

        return result


@routing_registry.register(
    "broadcast",
    display_name="广播路由",
    description="忽略拓扑结构，将 public_content 广播给所有其他 agent",
    config_schema={},
)
class BroadcastRoutingRegistered(BroadcastRouting):
    def __init__(self, **kwargs):
        super().__init__()
