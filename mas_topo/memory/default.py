"""默认记忆更新策略。

实现论文 Equation (3) 的精确版本：
H(t+1)_i ← H(t)_i ⊕ m(t)_pub,i ⊕ Σ_σ(t)({m(t)_priv,j | j ∈ N(t)_in(i)})

支持 DyTopo (semantic-private / spatial)、FT (ft-private / ft-public-up)
和 AutoGen GroupChat (broadcast / answer) 信道。
"""

from typing import Optional

from mas_topo._registries import memory_registry
from mas_topo.memory.base import MemoryUpdateStrategy
from mas_topo.message import AgentOutput, Message


@memory_registry.register(
    "default",
    display_name="默认记忆更新",
    description="论文 Equation (3) 精确实现：追加自身 public_content + 收到的 private 消息。",
)
class DefaultMemoryUpdate(MemoryUpdateStrategy):
    """默认记忆更新策略。

    - 仅追加自身 public_content 到自身记忆（m_pub,i）。
    - 仅追加沿邻接矩阵边收到的 private 消息（m_priv,j）。
    - 不做 public_content fallback（论文 Phase 4 精确实现）。
    - Manager 的 public_content 只用于自身记忆更新和 Phase 5 全局决策，
      不广播给 Worker。Worker 通过 context.goal（C_task）获得本轮目标。
    """

    # 支持的路由信道类型
    SUPPORTED_CHANNELS = (
        "semantic-private", "spatial", "ft-private", "ft-public-up",
        "broadcast", "answer",  # AutoGen GroupChat 全员广播信道
    )

    def update(
        self,
        agents: list,
        agent_names: list[str],
        routed_messages: dict[str, list[Message]],
        agent_outputs: Optional[dict[str, AgentOutput]] = None,
        round_num: int = 0,
        aggregation_order: Optional[list[int]] = None,
        context=None,
    ) -> None:
        _ = context  # 默认策略不使用上下文
        for agent in agents:
            private_msgs = []
            for msg in routed_messages.get(agent.name, []):
                if msg.channel in self.SUPPORTED_CHANNELS:
                    private_msgs.append(msg)

            if aggregation_order is not None and private_msgs:
                name_to_idx = {
                    name: idx for idx, name in enumerate(agent_names)
                }
                def _order_key(msg: Message):
                    sender_idx = name_to_idx.get(msg.sender_id, len(agent_names))
                    try:
                        pos = aggregation_order.index(sender_idx)
                    except ValueError:
                        pos = len(aggregation_order)
                    return pos
                private_msgs.sort(key=_order_key)

            if agent_outputs and agent.name in agent_outputs:
                output = agent_outputs[agent.name]
                own_content = output.public_content
                if own_content:
                    agent.append_memory(
                        f"[ROUND {round_num} - My Output]:\n{own_content}"
                    )

            for msg in private_msgs:
                agent.append_memory(
                    f"[From {msg.sender_id} ({msg.channel})]: {msg.content}"
                )
