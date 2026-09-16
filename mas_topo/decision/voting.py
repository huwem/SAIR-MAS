"""多数投票决策策略。

对所有 agent 的 public_content 做简单多数投票。
"""

from collections import Counter
from typing import Optional

from mas_topo._registries import decision_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.decision.base import DecisionStrategy, DecisionResult


class VotingDecision(DecisionStrategy):
    """多数投票决策策略。

    统计各 agent 的输出，取出现次数最多的作为最终答案。
    """

    def __init__(self, postprocessor: Optional[callable] = None, max_rounds: int = 10):
        self.postprocessor = postprocessor
        self.max_rounds = max_rounds

    def decide(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> DecisionResult:
        answers = []
        for name in agent_names:
            out = agent_outputs.get(name)
            if not out:
                continue
            text = out.to_text()
            if self.postprocessor:
                text = self.postprocessor(text)
            if text:
                answers.append(text.strip())

        if not answers:
            return DecisionResult(is_complete=False, next_goal="No answers yet.")

        counter = Counter(answers)
        winner, count = counter.most_common(1)[0]

        is_complete = context.round_num >= self.max_rounds or count >= 2

        return DecisionResult(
            is_complete=is_complete,
            next_goal="Converge on the most common answer.",
            final_answer=winner,
        )


@decision_registry.register(
    "voting",
    display_name="投票决策",
    description="对各 agent 的 public_content 做简单多数投票",
    config_schema={
        "max_rounds": {"type": "int", "default": 10, "description": "达到此轮次后强制完成"},
    },
)
class VotingDecisionRegistered(VotingDecision):
    def __init__(self, postprocessor: Optional[callable] = None,
                 max_rounds: int = 10, **kwargs):
        super().__init__(postprocessor=postprocessor, max_rounds=max_rounds)
