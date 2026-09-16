"""累计赞成票共识决策策略。

简单的累计投票机制：每个 Agent 的最新投票被记录，未激活的 Agent 维持原投票。
终止条件：True 票数 >= 已投票 Agent 半数（沉默未投票者不计入基数），且存在非空 answer。
"""

import logging
from typing import Optional

from mas_topo._registries import decision_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.decision.base import DecisionStrategy, DecisionResult

logger = logging.getLogger(__name__)


class ConsensusDecision(DecisionStrategy):
    """累计赞成票共识决策。

    每个 Agent 的最新投票被记录（True/False），未激活 Agent 保持旧值（不计入基数）。
    终止条件：
    - True 票数 >= 已投票 Agent 半数（沉默/未激活 Agent 不计入基数）
    - 存在非空 answer
    - 或达到 max_rounds 强制终止
    """

    def __init__(
        self,
        max_rounds: int = 20,
        required_answer_role: Optional[str] = None,
    ):
        self.max_rounds = max_rounds
        self.required_answer_role = required_answer_role
        # agent_name -> latest vote (bool), only agents that explicitly voted
        self._votes: dict[str, bool | None] = {}
        # agent_name -> latest non-empty answer
        self._answers: dict[str, str] = {}
        # agent_name -> round in which its latest answer was recorded
        self._answer_rounds: dict[str, int] = {}
        # agent_name -> round in which its latest vote was cast (v3 stance tracking)
        self._vote_rounds: dict[str, int] = {}
        # agent_name -> answer text at time of voting (v3 referent tracking)
        self._vote_answer_text: dict[str, str] = {}

    @staticmethod
    def _is_placeholder(out: AgentOutput) -> bool:
        """检测未激活 Agent 的空占位 AgentOutput。"""
        return (
            out.raw_response is None
            and out.vote_complete is None
            and out.is_complete is None
            and not out.answer
            and not out.public_content
            and not out.private_routing
        )

    def _current_answer_text(self) -> str:
        """团队最新 answer 文本（立场所指对象，v3 所指变更追踪）。"""
        best_round, best_text = -1, ""
        for name, text in self._answers.items():
            r = self._answer_rounds.get(name, 0)
            if r > best_round and str(text).strip():
                best_round, best_text = r, str(text)
        return best_text

    def _record_votes_and_answers(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> None:
        """Record votes and answers from activated agents (shared by subclasses)."""
        for name in agent_names:
            out = agent_outputs.get(name)
            if out and not self._is_placeholder(out):
                if out.answer and out.answer.strip():
                    self._answers[name] = out.answer.strip()
                    self._answer_rounds[name] = context.round_num
                vote = getattr(out, "vote_complete", None)
                if vote is None:
                    vote = out.is_complete
                if vote is not None:
                    self._votes[name] = bool(vote)
                    self._vote_rounds[name] = context.round_num
                    self._vote_answer_text[name] = self._current_answer_text()

    def decide(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> DecisionResult:
        """根据各 Agent 的 vote_complete 做出累计投票决策。"""
        self._record_votes_and_answers(agent_names, agent_outputs, context)

        # 第一轮投票设为 None：Agent 尚未看到彼此推理，投票不计数
        if context.round_num == 1:
            for name in self._votes:
                self._votes[name] = None

        logger.info("ConsensusDecision votes: %s answers: %s", self._votes, self._answers)

        true_count = sum(1 for v in self._votes.values() if v is True)

        # True 票数 >= 已投票 Agent 半数即终止（沉默 Agent 不计入基数）
        total_voters = len(self._votes)
        is_complete = total_voters > 0 and true_count >= total_voters / 2

        # 答案检查（使用跨轮持久化的 _answers）
        if is_complete:
            if self.required_answer_role:
                if not self._answers.get(self.required_answer_role, "").strip():
                    is_complete = False
            else:
                if not self._answers:
                    is_complete = False

        # 最大轮次强制终止
        if context.round_num >= self.max_rounds:
            is_complete = True

        # 提取最终答案
        final_answer = ""
        if is_complete:
            if self.required_answer_role and self.required_answer_role in self._answers:
                final_answer = self._answers[self.required_answer_role]
            else:
                # 取最新轮次记录的 answer；同轮时按名单逆序优先
                best_name = None
                best_round = -1
                for name in reversed(agent_names):
                    if name in self._answers:
                        r = self._answer_rounds.get(name, 0)
                        if r > best_round:
                            best_round = r
                            best_name = name
                if best_name is not None:
                    final_answer = self._answers[best_name]

        if is_complete:
            next_goal = "Task complete."
        else:
            true_names = [n for n, v in self._votes.items() if v is True]
            false_names = [n for n, v in self._votes.items() if v is False]
            silent_names = [n for n in agent_names if n not in self._votes]
            required = (total_voters + 1) // 2 if total_voters > 0 else 0
            parts = [
                f"Continue collaboration.",
                f"Approved ({len(true_names)}): {', '.join(true_names) if true_names else 'none'}.",
                f"Opposed ({len(false_names)}): {', '.join(false_names) if false_names else 'none'}.",
            ]
            if silent_names:
                parts.append(f"Abstained: {', '.join(silent_names)}.")
            parts.append(f"Need True >= {required} (half of {total_voters} voters).")
            next_goal = " ".join(parts)

        return DecisionResult(
            is_complete=is_complete,
            next_goal=next_goal,
            final_answer=final_answer,
            votes=dict(self._votes),
        )


@decision_registry.register(
    "consensus",
    display_name="累计赞成票共识决策",
    description="累计投票制：每个 Agent 最新投票被记录，True >= 已投票 Agent 半数时终止（未投票者不计入基数）。",
    config_schema={
        "max_rounds": {
            "type": "int",
            "default": 20,
            "description": "最大执行轮次（达到后强制完成）",
        },
        "required_answer_role": {
            "type": "str|null",
            "default": None,
            "description": "必须有非空 answer 产出才算完成",
        },
    },
)
class ConsensusDecisionRegistered(ConsensusDecision):
    """注册到全局注册表的版本，接受并忽略旧配置中的多余字段。"""

    def __init__(
        self,
        max_rounds: int = 20,
        required_answer_role: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            max_rounds=max_rounds,
            required_answer_role=required_answer_role,
        )
