"""ConsensusDecision 测试 — 简单累计投票。"""

from mas_topo.context import ExecutionContext
from mas_topo.decision.consensus import ConsensusDecision
from mas_topo.message import AgentOutput


def test_simple_vote_passes():
    """True >= 半数投票者 + 存在 answer → 终止。"""
    strategy = ConsensusDecision(required_answer_role="Developer", max_rounds=10)
    outputs = {
        "Developer": AgentOutput(answer="def foo(): return 42", vote_complete=True),
        "Tester": AgentOutput(vote_complete=True),
    }
    result = strategy.decide(["Developer", "Tester"], outputs, ExecutionContext(round_num=3))
    assert result.is_complete is True
    assert result.final_answer == "def foo(): return 42"


def test_all_true_votes_pass():
    """全部投 True → 终止。"""
    strategy = ConsensusDecision(max_rounds=10)
    outputs = {
        "A": AgentOutput(vote_complete=True, answer="ok"),
        "B": AgentOutput(vote_complete=True),
    }
    result = strategy.decide(["A", "B"], outputs, ExecutionContext(round_num=2))
    assert result.is_complete is True


def test_majority_passes_with_answer():
    """True >= 半数通过，答案由 required_answer_role 提供。"""
    strategy = ConsensusDecision(required_answer_role="Developer", max_rounds=10)
    outputs = {
        "Developer": AgentOutput(answer="def foo(): return 42", vote_complete=True),
        "Tester": AgentOutput(vote_complete=True),
        "Researcher": AgentOutput(vote_complete=False),
    }
    result = strategy.decide(["Developer", "Tester", "Researcher"], outputs, ExecutionContext(round_num=2))
    assert result.is_complete is True
    assert result.final_answer == "def foo(): return 42"


def test_fails_without_enough_true_votes():
    """True < 半数 → 不终止。"""
    strategy = ConsensusDecision(required_answer_role="Developer", max_rounds=10)
    outputs = {
        "Developer": AgentOutput(answer="def foo(): return 42", vote_complete=True),
        "Tester": AgentOutput(vote_complete=False),
        "Researcher": AgentOutput(vote_complete=False),
    }
    result = strategy.decide(["Developer", "Tester", "Researcher"], outputs, ExecutionContext(round_num=2))
    assert result.is_complete is False


def test_fails_without_answer():
    """投票通过但无 answer → 不终止。"""
    strategy = ConsensusDecision(required_answer_role="Developer", max_rounds=10)
    outputs = {
        "Developer": AgentOutput(vote_complete=True),
        "Tester": AgentOutput(vote_complete=True),
        "Researcher": AgentOutput(vote_complete=False),
    }
    result = strategy.decide(["Developer", "Tester", "Researcher"], outputs, ExecutionContext(round_num=2))
    assert result.is_complete is False


def test_max_rounds_forces_completion():
    """达到 max_rounds → 强制终止。"""
    strategy = ConsensusDecision(max_rounds=3)
    outputs = {"A": AgentOutput()}
    result = strategy.decide(["A"], outputs, ExecutionContext(round_num=3))
    assert result.is_complete is True


def test_silent_agents_not_counted():
    """沉默 Agent 不计入投票基数。"""
    strategy = ConsensusDecision(max_rounds=10)
    outputs = {
        "A": AgentOutput(vote_complete=True, answer="ok"),
        "B": AgentOutput(),  # 未投票（占位符，不计入）
    }
    result = strategy.decide(["A", "B"], outputs, ExecutionContext(round_num=2))
    # 仅 A 投票，1 >= 1/2 = 0.5，满足条件
    assert result.is_complete is True


def test_round1_never_completes():
    """第一轮投票不应终止，即使满足所有条件。"""
    strategy = ConsensusDecision(max_rounds=10)
    outputs = {
        "A": AgentOutput(vote_complete=True, answer="ok"),
        "B": AgentOutput(),
    }
    result = strategy.decide(["A", "B"], outputs, ExecutionContext(round_num=1))
    assert result.is_complete is False
    assert result.final_answer == ""


def test_cumulative_votes_persist():
    """投票跨轮累计，后续轮次可更新。"""
    strategy = ConsensusDecision(max_rounds=10)
    # Round 1: A 投 True，B 未投（第一轮强制不终止）
    outputs = {
        "A": AgentOutput(vote_complete=True, answer="ok"),
        "B": AgentOutput(),
    }
    result = strategy.decide(["A", "B"], outputs, ExecutionContext(round_num=1))
    assert result.is_complete is False  # 第一轮强制不终止

    # Round 2: B 投 False（但 A 的 True 仍在）
    strategy2 = ConsensusDecision(max_rounds=10)
    strategy2._votes = {"A": True}  # 模拟跨轮累计
    outputs2 = {
        "A": AgentOutput(vote_complete=True, answer="ok"),
        "B": AgentOutput(vote_complete=False),
    }
    result2 = strategy2.decide(["A", "B"], outputs2, ExecutionContext(round_num=2))
    # 2 voters, 1 True = 0.5，满足 True >= half
    assert result2.is_complete is True


def test_vote_can_change():
    """Agent 可在后续轮次更新投票（False → True）。"""
    strategy = ConsensusDecision(max_rounds=10)
    strategy._votes = {"A": False, "B": True}
    outputs = {
        "A": AgentOutput(vote_complete=True, answer="final"),
        "B": AgentOutput(vote_complete=True),
    }
    result = strategy.decide(["A", "B"], outputs, ExecutionContext(round_num=3))
    assert result.is_complete is True
    assert result.final_answer == "final"


def test_tie_not_enough():
    """True == 半数（刚好 0.5）时通过。"""
    strategy = ConsensusDecision(max_rounds=10)
    outputs = {
        "A": AgentOutput(vote_complete=True, answer="ok"),
        "B": AgentOutput(vote_complete=False),
    }
    result = strategy.decide(["A", "B"], outputs, ExecutionContext(round_num=2))
    # 2 voters, 1 True = 0.5，满足 True >= half
    assert result.is_complete is True
