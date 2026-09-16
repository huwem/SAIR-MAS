"""TLSG stance-commitment consensus (v3).

A vote is a stance commitment, not a tally entry.  Each agent's current
stance is its LATEST vote, indexed to the answer it evaluated.  A stance
leaves the count only via three events — never via time:

  1. replacement   — the agent casts a new vote (sincerity is present-tense);
  2. referent change — the team's newest answer text changed since the vote
     (the stance was about a DIFFERENT answer);
  3. abandonment   — False only: the objector was activated at least once
     after casting and did not reaffirm (informed silence = withdrawal).
     True stances need no maintenance: a motion on the table stands;
     only objections must be upheld.  This asymmetry is what converges.

Termination: count(True) > count(False) AND every voter in
``required_true_voters`` (e.g. the verification role) currently holds a
True stance AND a non-empty answer exists.
No health gate, no window, no time constant.  max_rounds is the safety valve.

Final answer extraction: the ROUND-START current answer (the value injected
into prompts, i.e. what agents actually evaluated and voted on this round),
NOT the freshest answer recorded during the completion round.  A same-round
answer change must not redirect an approval cast for the previous answer.
Falls back to the latest answer only when no round-start answer exists.
"""

import logging
from typing import Optional

from mas_topo._registries import decision_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.decision.base import DecisionResult
from mas_topo.decision.consensus import ConsensusDecision

logger = logging.getLogger(__name__)


class TlsgConsensus(ConsensusDecision):
    """立场承诺共识：True 数 > False 数且指定必须赞成者（如验证角色）立场为 True 时终止。"""

    def __init__(
        self,
        max_rounds: int = 20,
        required_answer_role: Optional[str] = None,
        required_true_voters: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            max_rounds=max_rounds,
            required_answer_role=required_answer_role,
        )
        # 必须投 True 的角色名单（如验证角色 Tester / Verifier）：
        # 其当前立场（失效事件处理后）必须为 True 才允许终止。
        self.required_true_voters = list(required_true_voters or [])

    # -- stance maintenance --------------------------------------------------

    def _invalidate_stances(self, agent_outputs: dict[str, AgentOutput],
                            current_round: int) -> None:
        """Apply three invalidation events to current stances."""
        current_answer = self._current_answer_text()

        for name in list(self._votes.keys()):
            # 2. referent change: answer text changed since vote was cast
            if self._vote_answer_text.get(name, "") != current_answer:
                logger.info("Stance by %s invalidated: referent changed.", name)
                del self._votes[name]
                continue

            # 3. abandonment (False only): activated after casting, silent now.
            #    Placeholder means the agent was given the floor but produced
            #    no real output — that is informed silence, objection lapses.
            if (
                self._votes[name] is False
                and self._vote_rounds.get(name, 0) < current_round
                and name in agent_outputs
            ):
                out = agent_outputs[name]
                vote = getattr(out, "vote_complete", None)
                if vote is None:
                    vote = out.is_complete
                if vote is None:
                    logger.info("Objection by %s withdrawn (silent after floor).", name)
                    del self._votes[name]

    # -- decide ---------------------------------------------------------------

    def decide(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> DecisionResult:
        # 轮首 current answer 快照：必须在 _record_votes_and_answers 之前取。
        # 本轮 Agent 投票评价的是注入 prompt 的这个旧值，收敛时也应提取它，
        # 而非本轮刚刷新的最新 answer（防止同轮改答案劫持投票结果）。
        round_start_answer = self._current_answer_text()

        # Record votes/answers (shared with parent) but NOT parent termination.
        self._record_votes_and_answers(agent_names, agent_outputs, context)

        # Safety valve: max_rounds forces completion.
        if context.round_num >= self.max_rounds:
            return DecisionResult(
                is_complete=True,
                next_goal="Task complete (max rounds).",
                final_answer=round_start_answer or self._extract_final_answer(agent_names),
                votes=dict(self._votes),
            )

        self._invalidate_stances(agent_outputs, context.round_num)

        # 第一轮投票设为 None：Agent 尚未看到彼此推理，投票不计数
        if context.round_num == 1:
            for name in list(self._votes.keys()):
                self._votes[name] = None

        true_count = sum(1 for v in self._votes.values() if v is True)
        false_count = sum(1 for v in self._votes.values() if v is False)

        # 必须赞成者检查：required_true_voters 中任一角色当前立场不为 True
        # （未投票 / 投 False / 立场已失效）则不允许终止。
        missing_required = [
            n for n in self.required_true_voters
            if self._votes.get(n) is not True
        ]
        is_complete = true_count > false_count and not missing_required
        if is_complete:
            if self.required_answer_role:
                is_complete = bool(self._answers.get(self.required_answer_role, "").strip())
            else:
                is_complete = bool(self._answers)

        if is_complete:
            metadata = getattr(context, "metadata", None)
            if metadata is not None:
                metadata.pop("tlsg_pending_objectors", None)
            return DecisionResult(
                is_complete=True,
                next_goal="Task complete.",
                final_answer=round_start_answer or self._extract_final_answer(agent_names),
                votes=dict(self._votes),
            )

        # Not complete: compute pending objectors for reflow activation.
        metadata = getattr(context, "metadata", None)
        if metadata is not None:
            if true_count > 0 and false_count > 0:
                metadata["tlsg_pending_objectors"] = {
                    n: self._vote_rounds.get(n, 0)
                    for n, v in self._votes.items()
                    if v is False
                }
            else:
                metadata.pop("tlsg_pending_objectors", None)

        return DecisionResult(
            is_complete=False,
            next_goal=self._stance_goal(true_count, false_count, missing_required),
            final_answer="",
            votes=dict(self._votes),
        )

    def _extract_final_answer(self, agent_names: list[str]) -> str:
        if self.required_answer_role and self.required_answer_role in self._answers:
            return self._answers[self.required_answer_role]
        best_name = None
        best_round = -1
        for name in reversed(agent_names):
            if name in self._answers:
                r = self._answer_rounds.get(name, 0)
                if r > best_round:
                    best_round = r
                    best_name = name
        return self._answers.get(best_name, "") if best_name else ""

    def _stance_goal(self, true_count: int, false_count: int,
                     missing_required: Optional[list[str]] = None) -> str:
        true_names = [n for n, v in self._votes.items() if v is True]
        false_names = [n for n, v in self._votes.items() if v is False]
        need = "Need True > False"
        if self.required_true_voters:
            need += f" and approval from: {', '.join(self.required_true_voters)}"
            if missing_required:
                need += f" (waiting: {', '.join(missing_required)})"
        need += "."
        return (
            f"Continue collaboration. Approved ({true_count}): "
            f"{', '.join(true_names) if true_names else 'none'}. "
            f"Opposed ({false_count}): {', '.join(false_names) if false_names else 'none'}. "
            f"{need}"
        )


@decision_registry.register(
    "tlsg_consensus",
    display_name="TLSG 立场承诺共识决策",
    description="立场承诺共识：每个 Agent 最新立场据改票/所指变更/弃议事件失效，"
    "True 数 > False 数且 answer 非空时终止。",
    config_schema={
        "max_rounds": {
            "type": "int",
            "default": 20,
            "description": "最大执行轮次（达到后强制完成）",
        },
        "required_answer_role": {
            "type": "str|null",
            "default": None,
            "description": "必须有非空 answer 产出的角色",
        },
        "required_true_voters": {
            "type": "list|null",
            "default": None,
            "description": "必须投 True 才允许终止的角色名单（如验证角色 Tester/Verifier）",
        },
    },
)
class TlsgConsensusRegistered(TlsgConsensus):
    """Registered variant for YAML-driven construction, accepts legacy kwargs silently."""

    def __init__(
        self,
        max_rounds: int = 20,
        required_answer_role: Optional[str] = None,
        required_true_voters: Optional[list[str]] = None,
        **kwargs,
    ):
        super().__init__(
            max_rounds=max_rounds,
            required_answer_role=required_answer_role,
            required_true_voters=required_true_voters,
        )
