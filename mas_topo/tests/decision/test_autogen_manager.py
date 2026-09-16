"""AutoGen GroupChatManager 决策策略与路由兜底单元测试。

对照 AutoGen v0.2 原版 groupchat.py 的核心机制：
- 提及计数（词边界 + 下划线变体）
- 发言人选择的重问与轮询兜底
- TERMINATE 令牌终止 + require_worker_answer 守卫
- 路由兜底：last_speaker 的下一个（next_agent 语义）
"""

from types import SimpleNamespace

from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.decision.autogen_manager import (
    AutoGenManagerDecision,
    mentioned_agents,
)
from mas_topo.routing.autogen_groupchat import AutoGenGroupChatRouting


def _make_llm_response(content, prompt_tokens=10, completion_tokens=5):
    return SimpleNamespace(
        content=content,
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        ),
    )


class FakeAdapter:
    """按队列返回预设回复的 llm_adapter。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def call(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return _make_llm_response(self.replies.pop(0))


def _make_manager(adapter):
    class _Mgr:
        def __init__(self):
            self.llm_adapter = adapter

        def get_memory_context(self):
            return "No prior context."

    return _Mgr()


def _context():
    ctx = ExecutionContext()
    ctx.task = "What is 2+2?"
    ctx.team_roster = [
        {"name": "Manager", "role": "Orchestrator"},
        {"name": "ProblemParser", "role": "Analyst"},
        {"name": "Solver", "role": "Solver"},
        {"name": "Verifier", "role": "Verifier"},
    ]
    return ctx


class TestMentionedAgents:
    def test_exactly_one(self):
        m = mentioned_agents("The next speaker should be Solver.", ["Solver", "Verifier"])
        assert m == {"Solver": 1}

    def test_multiple(self):
        m = mentioned_agents("Solver or Verifier?", ["Solver", "Verifier"])
        assert len(m) == 2

    def test_none(self):
        assert mentioned_agents("I don't know.", ["Solver", "Verifier"]) == {}

    def test_underscore_variants(self):
        # 键始终是 agent 原名，匹配支持下划线/空格/转义变体
        assert mentioned_agents("Story writer speaks.", ["Story_writer"]) == {"Story_writer": 1}
        assert mentioned_agents(r"Story\_writer speaks.", ["Story_writer"]) == {"Story_writer": 1}

    def test_word_boundary(self):
        # "SolverX" 不应匹配 "Solver"
        assert mentioned_agents("SolverX", ["Solver"]) == {}


class TestSpeakerSelection:
    def test_selects_single_mention(self):
        adapter = FakeAdapter(["Solver"])
        dec = AutoGenManagerDecision(manager_agent=_make_manager(adapter))
        outputs = {"ProblemParser": AgentOutput(public_content="analysis done")}
        result = dec.decide(["Manager", "ProblemParser", "Solver"], outputs, _context())
        assert not result.is_complete
        assert result.decision_output.next_speaker == "Solver"
        # 选择调用默认走 json_object 通道（selector_json_mode=True）
        assert adapter.calls[0][1].get("json_mode") is True

    def test_requery_on_multiple_mentions(self):
        adapter = FakeAdapter(["Solver or Verifier", "Verifier"])
        dec = AutoGenManagerDecision(manager_agent=_make_manager(adapter))
        outputs = {"Solver": AgentOutput(public_content="solution", answer="4")}
        result = dec.decide(["Manager", "Solver", "Verifier"], outputs, _context())
        assert result.decision_output.next_speaker == "Verifier"
        assert len(adapter.calls) == 2
        # 第二次调用的最后一条消息是 multiple 重问模板
        assert "more than one name" in adapter.calls[1][0][-1]["content"]

    def test_requery_on_no_mention(self):
        adapter = FakeAdapter(["whatever", "Solver"])
        dec = AutoGenManagerDecision(manager_agent=_make_manager(adapter))
        outputs = {"Verifier": AgentOutput(public_content="checked")}
        result = dec.decide(["Manager", "Solver", "Verifier"], outputs, _context())
        assert result.decision_output.next_speaker == "Solver"
        assert "didn't choose a speaker" in adapter.calls[1][0][-1]["content"]

    def test_fallback_next_after_last_speaker(self):
        # 全部尝试失败 → last_speaker 的下一个（原版 next_agent）
        adapter = FakeAdapter(["???", "???", "???"])
        dec = AutoGenManagerDecision(
            manager_agent=_make_manager(adapter), max_selection_retries=2,
        )
        outputs = {"ProblemParser": AgentOutput(public_content="done")}
        ctx = _context()
        result = dec.decide(
            ["Manager", "ProblemParser", "Solver", "Verifier"], outputs, ctx,
        )
        assert result.decision_output.next_speaker == "Solver"
        assert len(adapter.calls) == 3  # 1 + 2 retries

    def test_tokens_tracked(self):
        adapter = FakeAdapter(["Solver"])
        dec = AutoGenManagerDecision(manager_agent=_make_manager(adapter))
        ctx = _context()
        outputs = {"ProblemParser": AgentOutput(public_content="done")}
        dec.decide(["Manager", "ProblemParser", "Solver"], outputs, ctx)
        assert ctx.prompt_tokens == 10
        assert ctx.completion_tokens == 5

    def test_no_adapter_falls_back(self):
        dec = AutoGenManagerDecision(manager_agent=None, llm_adapter=None)
        outputs = {"Solver": AgentOutput(public_content="s")}
        result = dec.decide(["Manager", "Solver", "Verifier"], outputs, _context())
        assert result.decision_output.next_speaker == "Verifier"

    def test_llm_call_exception_falls_back(self):
        # 网关故障（adapter 重试耗尽后抛错）→ 降级为轮询，不让样本崩溃
        class RaisingAdapter(FakeAdapter):
            def call(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                raise RuntimeError("gateway junk response")

        adapter = RaisingAdapter([])
        dec = AutoGenManagerDecision(manager_agent=_make_manager(adapter))
        outputs = {"ProblemParser": AgentOutput(public_content="done")}
        result = dec.decide(
            ["Manager", "ProblemParser", "Solver", "Verifier"], outputs, _context(),
        )
        assert result.decision_output.next_speaker == "Solver"
        assert len(adapter.calls) == 1  # 基础设施故障立即兜底，不再重问


class TestSelectorJsonMode:
    def test_json_dict_reply(self):
        adapter = FakeAdapter(['{"role": "Solver"}'])
        dec = AutoGenManagerDecision(manager_agent=_make_manager(adapter))
        outputs = {"ProblemParser": AgentOutput(public_content="done")}
        result = dec.decide(
            ["Manager", "ProblemParser", "Solver", "Verifier"], outputs, _context(),
        )
        assert result.decision_output.next_speaker == "Solver"
        assert adapter.calls[0][1].get("json_mode") is True

    def test_json_string_reply(self):
        adapter = FakeAdapter(['"Verifier"'])
        dec = AutoGenManagerDecision(manager_agent=_make_manager(adapter))
        outputs = {"Solver": AgentOutput(public_content="done")}
        result = dec.decide(
            ["Manager", "Solver", "Verifier"], outputs, _context(),
        )
        assert result.decision_output.next_speaker == "Verifier"

    def test_json_unknown_name_falls_to_mention_count(self):
        # JSON 里的名字不在名册 → 忽略 JSON，回退提及计数（0 个 → 重问）
        adapter = FakeAdapter(['{"role": "Nobody"}', '{"role": "Solver"}'])
        dec = AutoGenManagerDecision(manager_agent=_make_manager(adapter))
        outputs = {"ProblemParser": AgentOutput(public_content="done")}
        result = dec.decide(
            ["Manager", "ProblemParser", "Solver", "Verifier"], outputs, _context(),
        )
        assert result.decision_output.next_speaker == "Solver"
        assert len(adapter.calls) == 2

    def test_json_mode_off_restores_plain_text(self):
        adapter = FakeAdapter(["Solver"])
        dec = AutoGenManagerDecision(
            manager_agent=_make_manager(adapter), selector_json_mode=False,
        )
        outputs = {"ProblemParser": AgentOutput(public_content="done")}
        result = dec.decide(
            ["Manager", "ProblemParser", "Solver", "Verifier"], outputs, _context(),
        )
        assert result.decision_output.next_speaker == "Solver"
        assert adapter.calls[0][1].get("json_mode") is False


class TestTerminateKeyword:
    def test_terminate_with_worker_answer_completes(self):
        adapter = FakeAdapter(["Solver"])
        dec = AutoGenManagerDecision(manager_agent=_make_manager(adapter))
        # 第 1 轮：Solver 产出 answer（无 TERMINATE）→ 不终止，做发言人选择
        r1 = dec.decide(
            ["Manager", "Solver", "Verifier"],
            {"Solver": AgentOutput(public_content="solution", answer="4")},
            _context(),
        )
        assert not r1.is_complete
        # 第 2 轮：Verifier 确认并给出 TERMINATE → 终止，answer 来自锁存
        result = dec.decide(
            ["Manager", "Solver", "Verifier"],
            {"Verifier": AgentOutput(public_content="The answer is correct. TERMINATE")},
            _context(),
        )
        assert result.is_complete
        assert result.final_answer == "4"
        # 终止轮不再做发言人选择（仅第 1 轮选择过 1 次）
        assert len(adapter.calls) == 1

    def test_terminate_blocked_without_worker_answer(self):
        adapter = FakeAdapter(["Solver"])
        dec = AutoGenManagerDecision(
            manager_agent=_make_manager(adapter), require_worker_answer=True,
        )
        outputs = {
            "Verifier": AgentOutput(public_content="looks done TERMINATE"),
        }
        result = dec.decide(["Manager", "Solver", "Verifier"], outputs, _context())
        assert not result.is_complete
        # 守卫拦截后仍做发言人选择
        assert len(adapter.calls) == 1


class TestRoutingFallback:
    """AutoGenGroupChatRouting 兜底：last_speaker 的下一个（next_agent 语义）。"""

    def test_fallback_next_of_last_speaker(self):
        routing = AutoGenGroupChatRouting(
            role_order=["ProblemParser", "Solver", "Verifier"],
        )
        prev = {"Solver": AgentOutput(public_content="solution")}
        active = routing.get_active_workers(
            ExecutionContext(), ["ProblemParser", "Solver", "Verifier"],
            previous_outputs=prev,
        )
        assert active == {"Verifier"}

    def test_fallback_wraps_around(self):
        routing = AutoGenGroupChatRouting(
            role_order=["ProblemParser", "Solver", "Verifier"],
        )
        prev = {"Verifier": AgentOutput(public_content="checked")}
        active = routing.get_active_workers(
            ExecutionContext(), ["ProblemParser", "Solver", "Verifier"],
            previous_outputs=prev,
        )
        assert active == {"ProblemParser"}

    def test_first_round_uses_role_order_head(self):
        routing = AutoGenGroupChatRouting(
            role_order=["ProblemParser", "Solver", "Verifier"],
        )
        active = routing.get_active_workers(
            ExecutionContext(), ["ProblemParser", "Solver", "Verifier"],
            previous_outputs=None,
        )
        assert active == {"ProblemParser"}

    def test_first_round_llm_selection(self):
        """首轮无历史时由 LLM 选择（原版 auto 语义），注入 llm_adapter 时生效。"""
        adapter = FakeAdapter(["Solver"])
        routing = AutoGenGroupChatRouting(
            role_order=["ProblemParser", "Solver", "Verifier"],
            llm_adapter=adapter,
        )
        ctx = ExecutionContext()
        ctx.task = "What is 2+2?"
        ctx.team_roster = [
            {"name": "ProblemParser", "role": "Analyst"},
            {"name": "Solver", "role": "Solver"},
            {"name": "Verifier", "role": "Verifier"},
        ]
        active = routing.get_active_workers(
            ctx, ["ProblemParser", "Solver", "Verifier"],
            previous_outputs=None,
        )
        assert active == {"Solver"}
        assert len(adapter.calls) == 1

    def test_manager_choice_wins(self):
        routing = AutoGenGroupChatRouting(
            role_order=["ProblemParser", "Solver", "Verifier"],
        )
        prev = {"Manager": AgentOutput(next_speaker="Verifier")}
        active = routing.get_active_workers(
            ExecutionContext(), ["ProblemParser", "Solver", "Verifier"],
            previous_outputs=prev,
        )
        assert active == {"Verifier"}

    def test_broadcast_ignores_adjacency(self):
        import numpy as np
        routing = AutoGenGroupChatRouting()
        outputs = {"Solver": AgentOutput(public_content="sol", answer="4")}
        names = ["Manager", "ProblemParser", "Solver", "Verifier"]
        adj = np.zeros((4, 4))  # 物理拓扑全断开也应广播
        routed = routing.route(names, outputs, adj, ExecutionContext())
        for dst in ("Manager", "ProblemParser", "Verifier"):
            assert len(routed[dst]) == 2  # public_content + answer
        assert routed["Solver"] == []
