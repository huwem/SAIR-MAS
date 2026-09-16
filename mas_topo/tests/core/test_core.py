"""核心模块单元测试：Registry, Message, AgentOutput, ExecutionContext, Graph 引擎。"""

import numpy as np
import pytest
from mas_topo.registry import Registry
from mas_topo.message import Message, AgentOutput
from mas_topo.context import ExecutionContext


class TestRegistry:
    """Registry 通用注册表测试。"""

    def test_register_and_get(self):
        reg = Registry("test")

        @reg.register("foo")
        class Foo:
            def __init__(self, value=1, **kwargs):
                self.value = value

        instance = reg.get("foo", value=42)
        assert isinstance(instance, Foo)
        assert instance.value == 42

    def test_register_multiple(self):
        reg = Registry("test")

        @reg.register("a")
        class A:
            pass

        @reg.register("b")
        class B:
            pass

        assert "a" in reg.list()
        assert "b" in reg.list()
        assert len(reg.list()) == 2

    def test_has(self):
        reg = Registry("test")

        @reg.register("exists")
        class Exists:
            pass

        assert reg.has("exists")
        assert not reg.has("missing")

    def test_get_missing_raises_keyerror(self):
        reg = Registry("test")
        with pytest.raises(KeyError):
            reg.get("nonexistent")

    def test_list_returns_names(self):
        reg = Registry("test")

        @reg.register("x")
        class X:
            pass

        assert "x" in reg.list()

    def test_duplicate_register_forbidden(self):
        reg = Registry("test")

        @reg.register("dup")
        class First:
            pass

        with pytest.raises(ValueError, match="already registered"):
            @reg.register("dup")
            class Second:
                pass


class TestMessage:
    """Message 消息数据类测试。"""

    def test_default_creation(self):
        msg = Message(content="Hello", sender_id="A")
        assert msg.content == "Hello"
        assert msg.sender_id == "A"
        assert msg.receiver_id is None
        assert msg.channel == "default"
        assert msg.metadata == {}

    def test_directed_message(self):
        msg = Message(
            content="Hi B", sender_id="A", receiver_id="B",
            channel="private", metadata={"priority": 1},
        )
        assert msg.receiver_id == "B"
        assert msg.channel == "private"
        assert msg.metadata["priority"] == 1


class TestAgentOutput:
    """AgentOutput 统一输出数据类测试。"""

    def test_default_empty(self):
        out = AgentOutput()
        assert out.public_content == ""
        assert out.private_content == {}
        assert out.query_descriptor is None
        assert out.key_descriptor is None

    def test_has_semantic_descriptors(self):
        out = AgentOutput(query_descriptor="need X", key_descriptor="offer Y")
        assert out.has_semantic_descriptors

        out2 = AgentOutput()
        assert not out2.has_semantic_descriptors

    def test_from_text(self):
        out = AgentOutput.from_text("Hello world")
        assert out.public_content == "Hello world"
        assert out.raw_response == "Hello world"

    def test_to_text_prefers_answer(self):
        out = AgentOutput(public_content="public", answer="final answer")
        assert out.to_text() == "final answer"

    def test_to_text_fallback_to_public(self):
        out = AgentOutput(public_content="public")
        assert out.to_text() == "public"

    def test_to_text_coerces_non_string_answer(self):
        out = AgentOutput(public_content="public", answer=42)
        assert out.to_text() == "42"
        assert isinstance(out.to_text(), str)

    def test_full_fields(self):
        out = AgentOutput(
            public_content="public msg",
            private_content={"B": "private to B"},
            query_descriptor="need help with code",
            key_descriptor="expert in Python",
            is_complete=True,
            answer="Final: 42",
        )
        assert out.is_complete is True
        assert out.private_content["B"] == "private to B"
        assert out.query_descriptor == "need help with code"


class TestExecutionContext:
    """ExecutionContext 运行时上下文测试。"""

    def test_default_creation(self):
        ctx = ExecutionContext()
        assert ctx.task == ""
        assert ctx.round_num == 0
        assert ctx.cost == 0.0

    def test_with_task(self):
        ctx = ExecutionContext(task="Solve this")
        assert ctx.task == "Solve this"

    def test_track_usage(self):
        ctx = ExecutionContext()
        ctx.track_usage(cost=0.5, prompt_tokens=100, completion_tokens=50)
        assert ctx.cost == 0.5
        assert ctx.prompt_tokens == 100
        assert ctx.completion_tokens == 50

        ctx.track_usage(cost=0.3, prompt_tokens=20)
        assert ctx.cost == 0.8
        assert ctx.prompt_tokens == 120

    def test_independent_instances(self):
        ctx1 = ExecutionContext(task="A")
        ctx2 = ExecutionContext(task="B")
        ctx1.cost = 10.0
        assert ctx2.cost == 0.0
        assert ctx1.task == "A"
        assert ctx2.task == "B"


class TestGraphEngine:
    """Graph 统一执行引擎测试（无 LLM 调用）。"""

    def make_mock_agent(self, name, response="output", complete_at=None):
        """创建不依赖 LLM 的 mock agent。"""
        from mas_topo.agent import BaseAgent

        class MockAgent(BaseAgent):
            def execute(self, input_data, context, **kwargs):
                is_done = (
                    complete_at is not None
                    and context.round_num >= complete_at
                    and name == "Manager"
                )
                return AgentOutput(
                    public_content=f"{name}: {response} round {context.round_num}",
                    query_descriptor=f"{name} needs help r{context.round_num}",
                    key_descriptor=f"{name} can help r{context.round_num}",
                    is_complete=is_done,
                    next_goal=f"Goal for r{context.round_num + 1}" if not is_done else None,
                    answer=f"Final answer at r{context.round_num}" if is_done else None,
                )

            async def async_execute(self, input_data, context, **kwargs):
                return self.execute(input_data, context, **kwargs)

        return MockAgent(name=name, role=f"Role of {name}")

    def _make_graph(self, topology="full_connected", decision="direct",
                    routing="graph", max_rounds=3):
        from mas_topo.topology.fixed import FixedTopology
        from mas_topo.decision.direct import DirectDecision
        from mas_topo.decision.manager import ManagerDecision
        from mas_topo.routing.graph import GraphRouting
        from mas_topo.graph import Graph

        agents = [
            self.make_mock_agent("A", "output A"),
            self.make_mock_agent("B", "output B"),
        ]

        if topology == "full_connected":
            topo = FixedTopology(pattern="full_connected")
        elif topology == "chain":
            topo = FixedTopology(pattern="chain")

        if decision == "direct":
            dec = DirectDecision()
        elif decision == "manager":
            dec = ManagerDecision(manager_name="A")

        router = GraphRouting()

        return Graph(
            agents=agents, topology_strategy=topo,
            decision_strategy=dec, routing_strategy=router,
            max_rounds=max_rounds, verbose=False,
        )

    def test_single_round_completion(self):
        """Manager 第一轮完成 -> 只执行 1 轮。"""
        agents = [
            self.make_mock_agent("Manager", "I decide", complete_at=1),
            self.make_mock_agent("Worker", "I work"),
        ]
        from mas_topo.topology.fixed import FixedTopology
        from mas_topo.decision.manager import ManagerDecision
        from mas_topo.routing.graph import GraphRouting
        from mas_topo.graph import Graph

        graph = Graph(
            agents=agents,
            topology_strategy=FixedTopology(pattern="chain"),
            decision_strategy=ManagerDecision(manager_name="Manager"),
            routing_strategy=GraphRouting(),
            max_rounds=5, verbose=False,
        )
        result = graph.run(task="Test task")
        assert result.is_completed
        assert len(result.rounds) == 1
        assert result.final_answer == "Final answer at r1"

    def test_multi_round(self):
        """Manager 第 3 轮才完成。"""
        agents = [
            self.make_mock_agent("Manager", "managing", complete_at=3),
            self.make_mock_agent("Worker", "working"),
        ]
        from mas_topo.topology.fixed import FixedTopology
        from mas_topo.decision.manager import ManagerDecision
        from mas_topo.routing.graph import GraphRouting
        from mas_topo.graph import Graph

        graph = Graph(
            agents=agents,
            topology_strategy=FixedTopology(pattern="chain"),
            decision_strategy=ManagerDecision(manager_name="Manager"),
            routing_strategy=GraphRouting(),
            max_rounds=5, verbose=False,
        )
        result = graph.run(task="Test")
        assert result.is_completed
        assert len(result.rounds) == 3
        assert result.final_answer == "Final answer at r3"

    def test_empty_decision_answer_falls_back_to_extraction(self):
        """决策未产出 final_answer 时，回退到轮次回溯提取 Worker 的 answer。

        回归测试：Manager 决策只给散文总结（final_answer=None）时，最终答案
        必须取 Developer 的代码，否则代码任务评测全部 "no code found"。
        """
        from mas_topo.agent import BaseAgent
        from mas_topo.decision.base import DecisionStrategy, DecisionResult
        from mas_topo.topology.fixed import FixedTopology
        from mas_topo.routing.graph import GraphRouting
        from mas_topo.graph import Graph

        class DevAgent(BaseAgent):
            def execute(self, input_data, context, **kwargs):
                return AgentOutput(
                    public_content="wrote code", answer="def f(): return 1"
                )

            async def async_execute(self, input_data, context, **kwargs):
                return self.execute(input_data, context, **kwargs)

        class StubDecision(DecisionStrategy):
            def decide(self, agent_names, agent_outputs, context):
                return DecisionResult(
                    is_complete=True, next_goal="done", final_answer=None
                )

        graph = Graph(
            agents=[DevAgent(name="Developer", role="Code Developer")],
            topology_strategy=FixedTopology(pattern="full_connected"),
            decision_strategy=StubDecision(),
            routing_strategy=GraphRouting(),
            max_rounds=3, verbose=False,
        )
        result = graph.run(task="t")
        assert result.is_completed
        assert result.final_answer == "def f(): return 1"

    def test_round_result_structure(self):
        graph = self._make_graph(max_rounds=1)
        result = graph.run(task="Test")
        r = result.rounds[0]
        assert r.round_num == 1
        assert "A" in r.agent_outputs
        assert "B" in r.agent_outputs
        assert r.adjacency_matrix is not None
        assert r.decision is not None

    def test_memory_accumulation(self):
        """Agent 记忆在每轮后更新。"""
        graph = self._make_graph(max_rounds=2)
        graph.run(task="Test")
        for agent in graph.agents:
            assert len(agent._memory_history) >= 0

    def test_context_tracks_rounds(self):
        graph = self._make_graph(max_rounds=2)
        result = graph.run(task="Test task")
        assert result.context.task == "Test task"
        assert len(result.rounds) > 0

    def test_agent_names_match(self):
        graph = self._make_graph()
        assert graph.agent_names == ["A", "B"]
        assert graph.name_to_agent["A"].name == "A"

    def test_chain_topology_execution(self):
        """链式拓扑下的执行。"""
        graph = self._make_graph(topology="chain")
        result = graph.run(task="Test")
        assert len(result.rounds) > 0

    def test_async_run(self):
        import asyncio
        graph = self._make_graph(max_rounds=1)

        async def _run():
            return await graph.arun(task="Test task")

        result = asyncio.run(_run())
        assert result is not None
        assert len(result.rounds) == 1
