"""Agent 模块单元测试。"""

import pytest

from mas_topo.agent import LLMAgent
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput


class TestLLMAgentParseResponse:
    """LLMAgent._parse_response 解析行为测试。"""

    def test_parse_empty_response_returns_raw_fallback(self):
        agent = LLMAgent(name="Developer", output_format="json")
        output = agent._parse_response("")
        assert output.raw_response == ""
        assert "empty" in output.public_content.lower() or "unparseable" in output.public_content.lower()

    def test_parse_invalid_json_returns_raw_fallback(self):
        agent = LLMAgent(name="Developer", output_format="json")
        output = agent._parse_response("not a json object")
        assert output.raw_response == "not a json object"
        assert output.public_content == "not a json object"[:500]

    def test_parse_valid_json_response(self):
        agent = LLMAgent(
            name="Developer",
            output_format="json",
            required_output_fields=["answer", "vote_complete"],
        )
        output = agent._parse_response(
            '{"answer": "def foo(): return 42", "vote_complete": true}'
        )
        assert output.answer == "def foo(): return 42"
        assert output.vote_complete is True

    def test_apply_output_defaults_sets_missing_optional_fields(self):
        output = AgentOutput()
        LLMAgent._apply_output_defaults(output)
        assert output.public_content == ""
        assert output.answer is None
        assert output.vote_complete is False
        assert output.private_routing == []
        assert output.tool_calls == []


class TestLLMAgentBareIO:
    """bare_io 裸调用模式：无 system prompt、user 即题目原文、输出全文即答案。"""

    def test_build_prompt_returns_task_only(self):
        from mas_topo.context import ExecutionContext
        agent = LLMAgent(name="Solver", bare_io=True)
        ctx = ExecutionContext()
        ctx.task = "What is 2+2?"
        ctx.round_num = 1
        system, user = agent._build_prompt(None, ctx)
        assert system == ""
        assert user == "What is 2+2?"

    def test_parse_response_maps_full_text_to_answer(self):
        agent = LLMAgent(name="Solver", bare_io=True)
        output = agent._parse_response("  The answer is 4.  ")
        assert output.answer == "The answer is 4."
        assert output.public_content == "The answer is 4."
        assert output.raw_response == "  The answer is 4.  "

    def test_chat_messages_omits_system_when_empty(self):
        assert LLMAgent._chat_messages("", "task") == [{"role": "user", "content": "task"}]
        msgs = LLMAgent._chat_messages("sys", "task")
        assert msgs[0] == {"role": "system", "content": "sys"}
        assert msgs[1] == {"role": "user", "content": "task"}


class TestLLMAgentTopologyPrompt:
    def _make_agent(self):
        return LLMAgent(
            name="Solver",
            required_output_fields=["private_routing"],
            local_neighbor_table={"Solver": {"ProblemParser", "Verifier"}},
        )

    def _make_context(self, topology_strategy: str):
        return ExecutionContext(
            task="Solve the problem",
            topology_strategy=topology_strategy,
            team_roster=[
                {"name": "ProblemParser"},
                {"name": "Solver"},
                {"name": "Verifier"},
            ],
        )

    def test_neighbor_topology_lists_neighbors_and_relay_hint_when_restricted(self):
        """拓扑受限（存在非邻居）时：注入邻居表和可达性/中转提示。"""
        agent = self._make_agent()
        context = self._make_context("neighbor")
        context.team_roster.append({"name": "Tester"})  # Tester 是 Solver 的非邻居

        _, user = agent._build_prompt(None, context)

        assert "Neighbors: ProblemParser, Verifier" in user
        assert "Only your neighbors are routable" in user
        assert "You may ask a neighbor to relay messages to non-neighbors." in user

    def test_neighbor_topology_full_connectivity_omits_neighbor_prompt(self):
        """全连通（无非邻居）时：不注入邻居相关提示词。"""
        agent = self._make_agent()

        _, user = agent._build_prompt(None, self._make_context("neighbor"))

        assert "Neighbors:" not in user
        assert "relay messages" not in user
        assert "routable" not in user

    def test_other_topology_still_omits_redundant_neighbors(self):
        agent = self._make_agent()

        _, user = agent._build_prompt(None, self._make_context("fixed"))

        assert "Neighbors:" not in user
        assert "relay messages" not in user

    def test_neighbor_topology_shows_none_for_isolated_agent(self):
        agent = self._make_agent()
        agent.local_neighbor_table["Solver"] = set()

        _, user = agent._build_prompt(None, self._make_context("neighbor"))

        assert "Neighbors: none" in user
        assert "relay messages" not in user
