"""决策策略模块单元测试。"""

from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.decision.base import DecisionResult
from mas_topo.decision.consensus import ConsensusDecision
from mas_topo.decision.direct import DirectDecision
from mas_topo.decision.llm_judge import LLMJudgeDecision
from mas_topo.decision.manager import ManagerDecision
from mas_topo.decision.voting import VotingDecision


class TestManagerDecision:
    """ManagerDecision 测试。"""

    def test_manager_complete(self):
        dec = ManagerDecision(manager_name="Manager")
        outputs = {
            "Manager": AgentOutput(
                is_complete=True, next_goal=None,
                answer="The answer is 42.",
            ),
            "Worker": AgentOutput(public_content="working"),
        }
        result = dec.decide(["Manager", "Worker"], outputs, ExecutionContext())
        assert result.is_complete
        assert result.final_answer == "The answer is 42."

    def test_manager_not_complete(self):
        dec = ManagerDecision(manager_name="Manager")
        outputs = {
            "Manager": AgentOutput(
                is_complete=False,
                next_goal="Refine the solution.",
            ),
        }
        result = dec.decide(["Manager"], outputs, ExecutionContext())
        assert not result.is_complete
        assert result.next_goal == "Refine the solution."

    def test_manager_missing(self):
        dec = ManagerDecision(manager_name="Manager")
        outputs = {"Worker": AgentOutput(public_content="work")}
        result = dec.decide(["Worker"], outputs, ExecutionContext())
        assert not result.is_complete
        assert "Continue" in result.next_goal

    def test_default_parameters(self):
        dec = ManagerDecision()
        assert dec.manager_name == "Manager"

    def test_require_worker_answer_blocks_early_completion(self):
        """require_worker_answer=True：无 Worker answer 时 Manager 的 is_complete 被拦截。"""
        dec = ManagerDecision(manager_name="Manager", require_worker_answer=True)
        outputs = {
            "Manager": AgentOutput(is_complete=True, next_goal="done", public_content="I solved it myself."),
            "Worker": AgentOutput(public_content="still analyzing"),
        }
        result = dec.decide(["Manager", "Worker"], outputs, ExecutionContext())
        assert not result.is_complete
        assert "no worker" in result.next_goal.lower() or "Continue" in result.next_goal

    def test_require_worker_answer_allows_completion_with_answer(self):
        """require_worker_answer=True：Worker 已有 answer 时 is_complete 正常生效。"""
        dec = ManagerDecision(manager_name="Manager", require_worker_answer=True)
        outputs = {
            "Manager": AgentOutput(is_complete=True, next_goal="done"),
            "Solver": AgentOutput(public_content="solution", answer="42"),
        }
        result = dec.decide(["Manager", "Solver"], outputs, ExecutionContext())
        assert result.is_complete

    def test_require_worker_answer_latches_across_rounds(self):
        """answer 锁存：上一轮出现过 answer 后，本轮 outputs 无 answer 也视为已产出。"""
        dec = ManagerDecision(manager_name="Manager", require_worker_answer=True)
        # 第 N 轮：Solver 产出 answer，Manager 不终止
        r1 = dec.decide(
            ["Manager", "Solver"],
            {"Manager": AgentOutput(is_complete=False, next_goal="verify"),
             "Solver": AgentOutput(public_content="solution", answer="42")},
            ExecutionContext(),
        )
        assert not r1.is_complete
        # 第 N+1 轮：Verifier 确认（无 answer 字段），Manager 终止
        r2 = dec.decide(
            ["Manager", "Verifier"],
            {"Manager": AgentOutput(is_complete=True, next_goal="done"),
             "Verifier": AgentOutput(public_content="confirmed correct")},
            ExecutionContext(),
        )
        assert r2.is_complete

    def test_require_worker_answer_default_off(self):
        """默认 False：无 Worker answer 时 Manager 的 is_complete 照常生效（旧行为不变）。"""
        dec = ManagerDecision(manager_name="Manager")
        outputs = {
            "Manager": AgentOutput(is_complete=True, next_goal="done", answer="42"),
            "Worker": AgentOutput(public_content="working"),
        }
        result = dec.decide(["Manager", "Worker"], outputs, ExecutionContext())
        assert result.is_complete

    def test_phase5_final_answer_is_none(self):
        """Phase 5 路径：Manager 只裁决 is_complete/next_goal，final_answer 必须为 None。

        最终解由 Graph 从轮次回溯提取 Solver/Developer 的 answer（论文 Algorithm 1
        Line 32）；若把 Manager 的 public_content（状态总结散文）当 final_answer
        返回，会被 Graph 当作最终答案，代码任务评测时报 "no code found"。
        """
        from unittest.mock import MagicMock
        from mas_topo.llm import LLMResponse

        mgr = MagicMock()
        mgr.system_prompt = "You are Manager."
        mgr.output_format = "json"
        mgr.output_schema = None
        mgr.llm_adapter.call.return_value = LLMResponse(content="{}")
        mgr._parse_json_response.return_value = AgentOutput(
            is_complete=True, next_goal="done",
            public_content="The task is complete. The solution works.",
        )
        dec = ManagerDecision(manager_name="Manager", manager_agent=mgr)
        outputs = {
            "Manager": AgentOutput(public_content="managing"),
            "Developer": AgentOutput(
                public_content="coded", answer="def f(): return 1"
            ),
        }
        result = dec.decide(["Manager", "Developer"], outputs, ExecutionContext())
        assert result.is_complete
        assert result.final_answer is None


class TestVotingDecision:
    """VotingDecision 测试。"""

    def test_majority_vote(self):
        dec = VotingDecision()
        outputs = {
            "A": AgentOutput(answer="Option X"),
            "B": AgentOutput(answer="Option X"),
            "C": AgentOutput(answer="Option Y"),
        }
        result = dec.decide(["A", "B", "C"], outputs, ExecutionContext(round_num=1))
        assert result.final_answer == "Option X"

    def test_no_answers(self):
        dec = VotingDecision()
        outputs = {"A": AgentOutput(), "B": AgentOutput()}
        result = dec.decide(["A", "B"], outputs, ExecutionContext())
        assert not result.is_complete

    def test_max_rounds_completion(self):
        dec = VotingDecision(max_rounds=3)
        outputs = {"A": AgentOutput(answer="X")}
        result = dec.decide(["A"], outputs, ExecutionContext(round_num=5))
        assert result.is_complete


class TestDirectDecision:
    """DirectDecision 测试。"""

    def test_takes_last_output(self):
        dec = DirectDecision()
        outputs = {
            "A": AgentOutput(public_content="first"),
            "B": AgentOutput(public_content="last"),
        }
        result = dec.decide(["A", "B"], outputs, ExecutionContext())
        assert "last" in result.final_answer

    def test_empty_outputs(self):
        dec = DirectDecision()
        result = dec.decide([], {}, ExecutionContext())
        assert result.final_answer == ""

    def test_int_answer_does_not_crash(self):
        """Regression: answer 为整数时 to_text() 返回字符串，不会触发 int.strip 错误。"""
        dec = DirectDecision()
        outputs = {
            "A": AgentOutput(public_content="first"),
            "B": AgentOutput(answer=42),
        }
        result = dec.decide(["A", "B"], outputs, ExecutionContext())
        assert result.final_answer == "42"
        assert result.is_complete

    def test_int_public_content_does_not_crash(self):
        """Regression: public_content 为整数时 to_text() 返回字符串。"""
        dec = DirectDecision()
        outputs = {
            "A": AgentOutput(public_content=42),
        }
        result = dec.decide(["A"], outputs, ExecutionContext())
        assert result.final_answer == "42"
        assert result.is_complete

    def test_falsy_int_answer_is_coerced_to_string(self):
        """answer=0 会被强制转为字符串 '0'，作为有效答案返回。"""
        dec = DirectDecision()
        outputs = {
            "A": AgentOutput(answer=0, public_content="fallback"),
        }
        result = dec.decide(["A"], outputs, ExecutionContext())
        assert result.final_answer == "0"

    def test_empty_answer_uses_public_content(self):
        """answer='' 时 to_text() 应回退到 public_content。"""
        dec = DirectDecision()
        outputs = {
            "A": AgentOutput(answer="", public_content="fallback"),
        }
        result = dec.decide(["A"], outputs, ExecutionContext())
        assert result.final_answer == "fallback"


class TestConsensusDecision:
    """ConsensusDecision 测试。"""

    def test_majority_int_answer_does_not_crash(self):
        """Regression: answer 为整数时 consensus 不应触发 int.strip 错误。"""
        dec = ConsensusDecision(
            required_answer_role="Solver",
            max_rounds=3,
        )
        outputs = {
            "Solver": AgentOutput(answer=42, vote_complete=True),
            "A": AgentOutput(vote_complete=True),
            "B": AgentOutput(vote_complete=True),
        }
        result = dec.decide(["Solver", "A", "B"], outputs, ExecutionContext(round_num=2))
        assert result.is_complete
        assert result.final_answer == "42"

    def test_round1_never_completes(self):
        """第一轮投票不应终止，即使全员赞成。"""
        dec = ConsensusDecision(
            required_answer_role="Solver",
            max_rounds=3,
        )
        outputs = {
            "Solver": AgentOutput(answer=42, vote_complete=True),
            "A": AgentOutput(vote_complete=True),
            "B": AgentOutput(vote_complete=True),
        }
        result = dec.decide(["Solver", "A", "B"], outputs, ExecutionContext(round_num=1))
        assert not result.is_complete
        assert result.final_answer == ""

    def test_zero_answer_not_treated_as_placeholder(self):
        """Regression: answer=0 是有效答案，不应被误判为空占位。"""
        dec = ConsensusDecision(
            required_answer_role="Solver",
            max_rounds=3,
        )
        outputs = {
            "Solver": AgentOutput(answer=0, vote_complete=True),
            "A": AgentOutput(vote_complete=True),
            "B": AgentOutput(vote_complete=True),
        }
        result = dec.decide(["Solver", "A", "B"], outputs, ExecutionContext(round_num=2))
        assert result.is_complete
        assert result.final_answer == "0"


class TestLLMJudgeDecision:
    """LLMJudgeDecision 测试。"""

    def test_fallback_without_llm(self):
        dec = LLMJudgeDecision(llm_adapter=None)
        outputs = {"A": AgentOutput(public_content="some answer")}
        result = dec.decide(["A"], outputs, ExecutionContext())
        assert result.final_answer == "some answer"

    def test_max_rounds(self):
        dec = LLMJudgeDecision(llm_adapter=None, max_rounds=5)
        outputs = {"A": AgentOutput(public_content="answer")}
        result = dec.decide(["A"], outputs, ExecutionContext(round_num=5))
        assert result.is_complete


class TestDecisionResult:
    """DecisionResult 测试。"""

    def test_default_fields(self):
        result = DecisionResult(is_complete=True)
        assert result.is_complete
        assert result.next_goal is None
        assert result.final_answer is None
        assert result.decision_output is None

    def test_full_fields(self):
        from mas_topo.message import AgentOutput
        out = AgentOutput(public_content="done")
        result = DecisionResult(
            is_complete=True, next_goal="Done.",
            final_answer="42", decision_output=out,
        )
        assert result.final_answer == "42"
        assert result.decision_output.public_content == "done"
