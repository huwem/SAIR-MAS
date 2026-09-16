"""LLM 裁决决策策略。

使用 LLM 汇总各 agent 输出并给出最终答案。
"""

from typing import Optional

from mas_topo._registries import decision_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.decision.base import DecisionStrategy, DecisionResult


class LLMJudgeDecision(DecisionStrategy):
    """LLM 裁决决策策略。

    使用独立的 LLM 调用汇总各 agent 输出，给出最终答案。
    """

    def __init__(
        self,
        llm_adapter: Optional[object] = None,
        system_prompt: str = "",
        max_rounds: int = 10,
    ):
        self.llm_adapter = llm_adapter
        self.system_prompt = system_prompt or (
            "You are the top decision-maker. "
            "Analyze and summarize all agents' outputs to give a final answer."
        )
        self.max_rounds = max_rounds

    def decide(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> DecisionResult:
        is_complete = context.round_num >= self.max_rounds

        if self.llm_adapter is None:
            first_out = next(iter(agent_outputs.values()), None)
            answer = first_out.to_text() if first_out else ""
            return DecisionResult(
                is_complete=is_complete or bool(answer),
                final_answer=answer,
                next_goal="Continue." if not answer else None,
            )

        parts = [f"Task: {context.task}\n\nAgent outputs:"]
        for name in agent_names:
            out = agent_outputs.get(name)
            if out:
                parts.append(
                    f"\n[{name}] ({out.query_descriptor or ''}):\n{out.public_content}"
                )
        parts.append("\nGive the final answer based on all agents' analysis.")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "\n".join(parts)},
        ]

        llm_response = self.llm_adapter.call(messages)
        context.track_usage(
            prompt_tokens=llm_response.usage.prompt_tokens,
            completion_tokens=llm_response.usage.completion_tokens,
        )
        return DecisionResult(
            is_complete=True,
            final_answer=llm_response.content,
            decision_output=AgentOutput(public_content=llm_response.content, raw_response=llm_response.content),
        )


@decision_registry.register(
    "llm_judge",
    display_name="LLM 裁决",
    description="使用独立 LLM 调用汇总各 agent 输出并给出最终答案",
    config_schema={
        "system_prompt": {"type": "str", "default": "", "description": "裁决提示词"},
        "max_rounds": {"type": "int", "default": 10, "description": "达到此轮次后强制完成"},
    },
)
class LLMJudgeDecisionRegistered(LLMJudgeDecision):
    def __init__(self, llm_adapter=None, system_prompt: str = "",
                 max_rounds: int = 10, **kwargs):
        super().__init__(llm_adapter=llm_adapter, system_prompt=system_prompt,
                         max_rounds=max_rounds)
