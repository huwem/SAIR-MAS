"""决策策略抽象基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from mas_topo.message import AgentOutput
from mas_topo.context import ExecutionContext


@dataclass
class DecisionResult:
    """决策结果。"""
    is_complete: bool
    next_goal: Optional[str] = None
    final_answer: Optional[str] = None
    decision_output: Optional[AgentOutput] = None
    votes: Optional[dict[str, bool]] = None


class DecisionStrategy(ABC):
    """决策策略抽象基类。"""

    @abstractmethod
    def decide(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> DecisionResult:
        """根据各 Agent 输出做出决策。"""
