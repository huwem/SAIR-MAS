"""记忆更新策略抽象基类。

定义 MemoryUpdateStrategy 接口，支持通过配置选择不同的记忆更新方式。
不同路由策略可配合不同的记忆更新策略使用。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from mas_topo.message import AgentOutput, Message


class MemoryUpdateStrategy(ABC):
    """记忆更新策略抽象基类。

    实现论文 Equation (3) 的不同变体：
    H(t+1)_i ← H(t)_i ⊕ m(t)_pub,i ⊕ Σ_σ(t)({m(t)_priv,j | j ∈ N(t)_in(i)})
    """

    @abstractmethod
    def update(
        self,
        agents: list,
        agent_names: list[str],
        routed_messages: dict[str, list[Message]],
        agent_outputs: Optional[dict[str, AgentOutput]] = None,
        round_num: int = 0,
        aggregation_order: Optional[list[int]] = None,
        context: Optional[Any] = None,
    ) -> None:
        """更新所有 Agent 的记忆。

        Args:
            agents: Agent 实例列表。
            agent_names: Agent 名称列表（与 agents 一一对应）。
            routed_messages: 路由后的消息，按 receiver_id 分组。
            agent_outputs: 本轮 Agent 输出。
            round_num: 当前轮次。
            aggregation_order: 聚合顺序（拓扑策略提供）。
            context: 运行时上下文（P2P 策略用于访问共享 ledger）。
        """
