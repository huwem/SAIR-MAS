"""决策策略模块。

提供多种决策策略：
- ManagerDecision: Manager agent 决策
- VotingDecision: 多数投票决策
- DirectDecision: 直通决策（取最后 agent 输出）
- LLMJudgeDecision: LLM 裁决决策
"""

from mas_topo.decision.base import DecisionStrategy, DecisionResult
from mas_topo.decision.manager import ManagerDecision
from mas_topo.decision.voting import VotingDecision
from mas_topo.decision.direct import DirectDecision
from mas_topo.decision.llm_judge import LLMJudgeDecision
from mas_topo.decision.consensus import ConsensusDecision
from mas_topo.decision.tlsg_consensus import TlsgConsensus
from mas_topo.decision.autogen_manager import AutoGenManagerDecision

__all__ = [
    "DecisionStrategy",
    "DecisionResult",
    "ManagerDecision",
    "VotingDecision",
    "DirectDecision",
    "LLMJudgeDecision",
    "ConsensusDecision",
    "TlsgConsensus",
    "AutoGenManagerDecision",
]
