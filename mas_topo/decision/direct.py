"""直通决策策略。

直接取指定答案角色（或最后一个 agent）的输出作为最终答案，达到 max_rounds 时完成。
"""

from typing import Optional

from mas_topo._registries import decision_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.decision.base import DecisionStrategy, DecisionResult


class DirectDecision(DecisionStrategy):
    """直通决策策略。

    直接取指定答案角色（required_answer_role）的输出作为最终答案；
    若未指定，则回退到 agent_names 中最后一个有非空输出的 agent。
    终止条件：达到 max_rounds，或答案角色显式输出 is_complete=true。
    """

    def __init__(
        self,
        max_rounds: int = 10,
        required_answer_role: Optional[str] = None,
    ):
        self.max_rounds = max_rounds
        self.required_answer_role = required_answer_role

    def decide(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> DecisionResult:
        last_output = None

        # 优先取指定答案角色的输出
        if self.required_answer_role:
            required_out = agent_outputs.get(self.required_answer_role)
            if required_out and required_out.to_text().strip():
                last_output = required_out

        # 未指定或指定角色无输出时，回退到最后一个有非空输出的 agent
        if last_output is None:
            for name in reversed(agent_names):
                out = agent_outputs.get(name)
                if out and out.to_text().strip():
                    last_output = out
                    break

        final_answer = last_output.to_text() if last_output else ""

        # 优先读取 agent 输出的 is_complete 字段，支持单智能体多轮自迭代
        if last_output is not None and last_output.is_complete is not None:
            is_complete = last_output.is_complete
        elif self.required_answer_role:
            # 指定了答案角色时，仅在达到 max_rounds 时终止，
            # 避免任意非空输出导致提前结束
            is_complete = context.round_num >= self.max_rounds
        else:
            # 未指定答案角色时保持旧行为：非空输出即视为完成
            is_complete = context.round_num >= self.max_rounds or bool(final_answer)

        return DecisionResult(
            is_complete=is_complete,
            next_goal="Continue.",
            final_answer=final_answer,
            decision_output=last_output,
        )


@decision_registry.register(
    "direct",
    display_name="直通决策",
    description="直接取指定答案角色（或最后一个 agent）输出作为最终答案，达到 max_rounds 时完成",
    config_schema={
        "max_rounds": {"type": "int", "default": 10, "description": "达到此轮次后强制完成"},
        "required_answer_role": {"type": "str|null", "default": None, "description": "指定最终答案来源角色"},
    },
)
class DirectDecisionRegistered(DirectDecision):
    def __init__(
        self,
        max_rounds: int = 10,
        required_answer_role: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(max_rounds=max_rounds, required_answer_role=required_answer_role)
