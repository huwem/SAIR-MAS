import pytest

from mas_topo.context import ExecutionContext
from mas_topo.ledger import IllocutionaryLedger
from mas_topo.memory.ledger_memory import LedgerMemoryUpdate, LEDGER_SUMMARIES_KEY
from mas_topo.message import AgentOutput, Channel, Message
from mas_topo.agent import LLMAgent


class FakeAgent:
    def __init__(self, name):
        self.name = name
        self.memory = []

    def append_memory(self, content, round_num=0):
        self.memory.append(content)


def test_ledger_memory_injects_summary_to_context_metadata():
    ledger = IllocutionaryLedger()
    ledger.append(1, "A", ["B"], "Directive", "do X")
    strategy = LedgerMemoryUpdate(inject_ledger_summary="always")
    agent_b = FakeAgent("B")
    context = ExecutionContext(round_num=1)
    context.ledger = ledger
    strategy.update(
        agents=[agent_b],
        agent_names=["A", "B"],
        routed_messages={"B": [Message(content="[Directive]: do X", sender_id="A", receiver_id="B", channel=Channel.FT_PRIVATE)]},
        agent_outputs={"B": AgentOutput(
            private_routing=[{"recipients": ["A"], "illocution": "Assertive", "content": "got it"}],
        )},
        context=context,
        round_num=1,
    )
    # Ledger 摘要不应在 memory 中
    assert not any("Global Contract Ledger Summary" in m for m in agent_b.memory)
    # 应在 context.metadata 的 per-agent 摘要字典中
    summaries = context.metadata.get(LEDGER_SUMMARIES_KEY, {})
    assert "B" in summaries
    assert "Global Contract Ledger Summary" in summaries["B"]
    # 私信仍在 memory 中
    assert any("[From A (private)]" in m for m in agent_b.memory)
    # 自己发出的 private_routing 记入记忆
    assert any("I sent to A" in m for m in agent_b.memory)


def test_ledger_memory_no_summary_when_empty():
    ledger = IllocutionaryLedger()
    strategy = LedgerMemoryUpdate()
    agent_c = FakeAgent("C")
    context = ExecutionContext(round_num=1)
    context.ledger = ledger
    strategy.update(
        agents=[agent_c],
        agent_names=["A", "B", "C"],
        routed_messages={},
        agent_outputs={},
        context=context,
        round_num=1,
    )
    summaries = context.metadata.get(LEDGER_SUMMARIES_KEY, {})
    assert "C" not in summaries


def test_ledger_memory_updates_context_current_answer():
    ledger = IllocutionaryLedger()
    ledger.append(
        1, "Developer", ["Tester"], "Assertive", "Here is the code",
        answer_snapshot="def foo(): return 42",
    )
    strategy = LedgerMemoryUpdate()
    agent_tester = FakeAgent("Tester")
    context = ExecutionContext(round_num=1)
    context.ledger = ledger
    strategy.update(
        agents=[agent_tester],
        agent_names=["Developer", "Tester"],
        routed_messages={},
        agent_outputs={},
        context=context,
        round_num=1,
    )
    # answer 不写入记忆，只通过 context.metadata 动态注入
    assert not any("Current Best Answer" in m for m in agent_tester.memory)
    assert not any("def foo(): return 42" in m for m in agent_tester.memory)
    assert context.metadata.get("current_answer") == "def foo(): return 42"


def test_ledger_memory_uses_latest_answer_snapshot():
    ledger = IllocutionaryLedger()
    ledger.append(
        1, "Developer", ["Tester"], "Assertive", "first",
        answer_snapshot="def v1(): pass",
    )
    ledger.append(
        2, "Developer", ["Tester"], "Assertive", "second",
        answer_snapshot="def v2(): pass",
    )
    strategy = LedgerMemoryUpdate()
    agent_tester = FakeAgent("Tester")
    context = ExecutionContext(round_num=2)
    context.ledger = ledger
    strategy.update(
        agents=[agent_tester],
        agent_names=["Developer", "Tester"],
        routed_messages={},
        agent_outputs={},
        context=context,
        round_num=2,
    )
    # 记忆中不应残留旧 answer
    assert not any("def v1(): pass" in m for m in agent_tester.memory)
    assert not any("def v2(): pass" in m for m in agent_tester.memory)
    assert context.metadata.get("current_answer") == "def v2(): pass"


def test_agent_prompt_injects_current_answer_via_context_fields():
    agent = LLMAgent(
        name="Tester",
        context_fields=["current_answer"],
    )
    context = ExecutionContext(task="test task", round_num=1)
    context.metadata = {"current_answer": "def foo(): return 42"}
    _, user_prompt = agent._build_prompt("test task", context)
    assert "[current_answer]" in user_prompt
    assert "def foo(): return 42" in user_prompt


def test_agent_prompt_labels_current_answer_with_source():
    """current_answer 带来源时，注入标签改为 answer_from_<来源角色>。"""
    agent = LLMAgent(
        name="Tester",
        context_fields=["current_answer"],
    )
    context = ExecutionContext(task="test task", round_num=1)
    context.metadata = {
        "current_answer": "def foo(): return 42",
        "current_answer_source": "Developer",
    }
    _, user_prompt = agent._build_prompt("test task", context)
    assert "[answer_from_developer]" in user_prompt
    assert "[current_answer]" not in user_prompt
    assert "def foo(): return 42" in user_prompt


def test_ledger_memory_update_validates_inject_ledger_summary():
    with pytest.raises(ValueError):
        LedgerMemoryUpdate(inject_ledger_summary="sometimes")


def test_ledger_memory_update_validates_tool_memory_mode():
    with pytest.raises(ValueError):
        LedgerMemoryUpdate(tool_memory_mode="full")


def test_ledger_memory_update_defaults():
    strategy = LedgerMemoryUpdate()
    assert strategy.inject_ledger_summary == "abnormal_only"
    assert strategy.tool_memory_mode == "result_only"


def test_ledger_summary_abnormal_only_empty_when_normal():
    ledger = IllocutionaryLedger()
    ledger.append(1, "A", ["B"], "Directive", "do X")
    strategy = LedgerMemoryUpdate(inject_ledger_summary="abnormal_only")
    agent_b = FakeAgent("B")
    context = ExecutionContext(round_num=1)
    context.ledger = ledger
    strategy.update(
        agents=[agent_b],
        agent_names=["A", "B"],
        routed_messages={},
        agent_outputs={},
        context=context,
        round_num=1,
    )
    summaries = context.metadata.get("__ledger_summaries__", {})
    assert summaries.get("B", "") == ""


def test_ledger_summary_abnormal_only_injects_when_disputed():
    ledger = IllocutionaryLedger()
    entry = ledger.append(1, "A", ["B"], "Directive", "do X")
    ledger.update_status(entry.entry_id, "DISPUTED")
    strategy = LedgerMemoryUpdate(inject_ledger_summary="abnormal_only")
    agent_b = FakeAgent("B")
    context = ExecutionContext(round_num=1)
    context.ledger = ledger
    strategy.update(
        agents=[agent_b],
        agent_names=["A", "B"],
        routed_messages={},
        agent_outputs={},
        context=context,
        round_num=1,
    )
    summaries = context.metadata.get("__ledger_summaries__", {})
    assert "B" in summaries
    assert "DISPUTED" in summaries["B"]


class FakeOutput:
    def __init__(self, tool_calls=None, private_routing=None):
        self.tool_calls = tool_calls or []
        self.private_routing = private_routing or []


def test_tool_memory_result_only_omits_args():
    strategy = LedgerMemoryUpdate(tool_memory_mode="result_only")
    agent = FakeAgent("Tester")
    context = ExecutionContext(round_num=1)
    strategy.update(
        agents=[agent],
        agent_names=["Tester"],
        routed_messages={},
        agent_outputs={
            "Tester": FakeOutput(
                tool_calls=[
                    {
                        "call": {"name": "run_python", "arguments": {"code": "very long code"}},
                        "result": {"result": {"passed": True, "stderr": ""}},
                    }
                ]
            )
        },
        context=context,
        round_num=1,
    )
    assert any("tool run_python" in m for m in agent.memory)
    assert not any("very long code" in m for m in agent.memory)
    assert any("passed=True" in m for m in agent.memory)


def test_tool_memory_result_only_includes_failure_stderr():
    strategy = LedgerMemoryUpdate(tool_memory_mode="result_only")
    agent = FakeAgent("Tester")
    context = ExecutionContext(round_num=1)
    strategy.update(
        agents=[agent],
        agent_names=["Tester"],
        routed_messages={},
        agent_outputs={
            "Tester": FakeOutput(
                tool_calls=[
                    {
                        "call": {"name": "run_python", "arguments": {"code": "x"}},
                        "result": {"result": {"passed": False, "stderr": "AssertionError: expected 5 but got 3"}},
                    }
                ]
            )
        },
        context=context,
        round_num=1,
    )
    assert any("passed=False" in m for m in agent.memory)
    assert any("AssertionError" in m for m in agent.memory)


def test_tool_memory_none_does_not_record():
    strategy = LedgerMemoryUpdate(tool_memory_mode="none")
    agent = FakeAgent("Tester")
    context = ExecutionContext(round_num=1)
    strategy.update(
        agents=[agent],
        agent_names=["Tester"],
        routed_messages={},
        agent_outputs={
            "Tester": FakeOutput(
                tool_calls=[
                    {
                        "call": {"name": "run_python", "arguments": {"code": "x"}},
                        "result": {"result": {"passed": True}},
                    }
                ]
            )
        },
        context=context,
        round_num=1,
    )
    assert not any("Tool run_python" in m for m in agent.memory)


def test_tool_memory_verbose_keeps_args():
    strategy = LedgerMemoryUpdate(tool_memory_mode="verbose")
    agent = FakeAgent("Tester")
    context = ExecutionContext(round_num=1)
    strategy.update(
        agents=[agent],
        agent_names=["Tester"],
        routed_messages={},
        agent_outputs={
            "Tester": FakeOutput(
                tool_calls=[
                    {
                        "call": {"name": "run_python", "arguments": {"code": "long code here"}},
                        "result": {"result": {"passed": True, "stderr": ""}},
                    }
                ]
            )
        },
        context=context,
        round_num=1,
    )
    assert any("long code here" in m for m in agent.memory)
    assert any("args=" in m for m in agent.memory)


def test_tool_memory_summary_includes_truncated_args():
    """summary 模式：记录截断后的输入参数 + 结果，不泄露完整代码。"""
    strategy = LedgerMemoryUpdate(tool_memory_mode="summary")
    agent = FakeAgent("Tester")
    context = ExecutionContext(round_num=1)
    long_code = "x = 1\n" * 100 + "\nUNIQUE_TAIL_MARKER"
    strategy.update(
        agents=[agent],
        agent_names=["Tester"],
        routed_messages={},
        agent_outputs={
            "Tester": FakeOutput(
                tool_calls=[
                    {
                        "call": {"name": "run_python", "arguments": {"code": long_code}},
                        "result": {"result": {"passed": True, "stderr": ""}},
                    }
                ]
            )
        },
        context=context,
        round_num=1,
    )
    assert any("args=" in m and "passed=True" in m for m in agent.memory)
    assert not any("UNIQUE_TAIL_MARKER" in m for m in agent.memory)


def test_ledger_summary_never_returns_empty():
    ledger = IllocutionaryLedger()
    ledger.append(1, "A", ["B"], "Directive", "do X", status="DISPUTED")
    strategy = LedgerMemoryUpdate(inject_ledger_summary="never")
    agent_b = FakeAgent("B")
    context = ExecutionContext(round_num=1)
    context.ledger = ledger
    strategy.update(
        agents=[agent_b],
        agent_names=["A", "B"],
        routed_messages={},
        agent_outputs={},
        context=context,
        round_num=1,
    )
    summaries = context.metadata.get("__ledger_summaries__", {})
    assert summaries.get("B", "") == ""
