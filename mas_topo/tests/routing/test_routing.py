"""路由策略模块单元测试。"""

import numpy as np
from mas_topo.context import ExecutionContext
from mas_topo.message import Message, AgentOutput
from mas_topo.routing.graph import GraphRouting
from mas_topo.routing.semantic import SemanticRouting
from mas_topo.routing.broadcast import BroadcastRouting


class TestGraphRouting:
    """GraphRouting 图邻接路由测试。"""

    def test_basic_routing(self):
        router = GraphRouting()
        adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        outputs = {
            "A": AgentOutput(public_content="msg from A"),
            "B": AgentOutput(public_content="msg from B"),
            "C": AgentOutput(public_content="msg from C"),
        }
        result = router.route(["A", "B", "C"], outputs, adj, ExecutionContext())

        assert len(result["B"]) == 1
        assert result["B"][0].sender_id == "A"
        assert result["B"][0].content == "msg from A"

        assert len(result["C"]) == 1
        assert result["C"][0].sender_id == "B"

        assert len(result["A"]) == 0

    def test_no_edges(self):
        router = GraphRouting()
        adj = np.zeros((3, 3), dtype=int)
        outputs = {
            "A": AgentOutput(public_content="msg"),
            "B": AgentOutput(public_content="msg"),
            "C": AgentOutput(public_content="msg"),
        }
        result = router.route(["A", "B", "C"], outputs, adj, ExecutionContext())
        assert all(len(msgs) == 0 for msgs in result.values())

    def test_full_connected(self):
        router = GraphRouting()
        N = 3
        adj = np.ones((N, N), dtype=int)
        np.fill_diagonal(adj, 0)
        outputs = {f"Agent{i}": AgentOutput(public_content=f"msg{i}") for i in range(N)}
        result = router.route([f"Agent{i}" for i in range(N)], outputs, adj, ExecutionContext())

        for name in [f"Agent{i}" for i in range(N)]:
            assert len(result[name]) == N - 1


class TestSemanticRouting:
    """SemanticRouting 语义路由测试。"""

    def test_private_content_priority(self):
        router = SemanticRouting(memory_truncate=500)
        adj = np.array([[0, 1], [0, 0]])
        outputs = {
            "A": AgentOutput(
                public_content="public A",
                private_content={"B": "private to B"},
            ),
            "B": AgentOutput(public_content="public B"),
        }
        result = router.route(["A", "B"], outputs, adj, ExecutionContext())

        assert len(result["B"]) == 1
        assert result["B"][0].content == "private to B"

    def test_no_private_content_no_message(self):
        """无 private_content 时不路由任何消息（不做 public_content fallback）。"""
        router = SemanticRouting()
        adj = np.array([[0, 1], [0, 0]])
        outputs = {
            "A": AgentOutput(public_content="public A", answer="some code"),
            "B": AgentOutput(public_content="public B"),
        }
        result = router.route(["A", "B"], outputs, adj, ExecutionContext())
        assert len(result["B"]) == 0

    def test_answer_appended_to_private_message(self):
        """private_content 非空时，附带 sender 的 answer 字段。"""
        router = SemanticRouting()
        adj = np.array([[0, 1], [0, 0]])
        outputs = {
            "A": AgentOutput(
                private_content={"B": "Please test this code"},
                answer="def foo(): return 42",
            ),
            "B": AgentOutput(),
        }
        result = router.route(["A", "B"], outputs, adj, ExecutionContext())
        assert len(result["B"]) == 1
        msg = result["B"][0]
        assert "Please test this code" in msg.content
        assert "def foo(): return 42" in msg.content
        assert "[Answer]" in msg.content

    def test_memory_truncate(self):
        router = SemanticRouting(memory_truncate=10)
        adj = np.array([[0, 1], [0, 0]])
        outputs = {
            "A": AgentOutput(private_content={"B": "This is a very long message that should be truncated"}),
            "B": AgentOutput(),
        }
        result = router.route(["A", "B"], outputs, adj, ExecutionContext())

        assert len(result["B"]) == 1
        assert len(result["B"][0].content) <= 10

    def test_metadata_includes_descriptors(self):
        router = SemanticRouting()
        adj = np.array([[0, 1], [0, 0]])
        outputs = {
            "A": AgentOutput(
                private_content={"B": "private msg"},
                query_descriptor="need X",
                key_descriptor="offer Y",
            ),
            "B": AgentOutput(),
        }
        result = router.route(["A", "B"], outputs, adj, ExecutionContext())

        assert result["B"][0].metadata["query_descriptor"] == "need X"
        assert result["B"][0].metadata["key_descriptor"] == "offer Y"


class TestBroadcastRouting:
    """BroadcastRouting 广播路由测试。"""

    def test_broadcast_to_all(self):
        router = BroadcastRouting()
        adj = np.zeros((3, 3))  # 忽略拓扑
        outputs = {
            "A": AgentOutput(public_content="msg A"),
            "B": AgentOutput(public_content="msg B"),
            "C": AgentOutput(public_content="msg C"),
        }
        result = router.route(["A", "B", "C"], outputs, adj, ExecutionContext())

        # 每个 agent 应该收到其他 2 个 agent 的消息
        for name in ["A", "B", "C"]:
            assert len(result[name]) == 2
            senders = {m.sender_id for m in result[name]}
            assert name not in senders  # 不包含自己

    def test_ignore_adjacency(self):
        """广播路由不应受邻接矩阵影响。"""
        router = BroadcastRouting()
        adj_empty = np.zeros((2, 2))
        adj_full = np.ones((2, 2))
        outputs = {
            "A": AgentOutput(public_content="msg"),
            "B": AgentOutput(public_content="msg"),
        }
        result_empty = router.route(["A", "B"], outputs, adj_empty, ExecutionContext())
        result_full = router.route(["A", "B"], outputs, adj_full, ExecutionContext())

        assert len(result_empty["A"]) == len(result_full["A"])
        assert len(result_empty["B"]) == len(result_full["B"])

    def test_empty_content_skipped(self):
        router = BroadcastRouting()
        adj = np.zeros((2, 2))
        outputs = {
            "A": AgentOutput(public_content=""),
            "B": AgentOutput(public_content="msg B"),
        }
        result = router.route(["A", "B"], outputs, adj, ExecutionContext())
        assert len(result["B"]) == 0  # A 无内容
        assert len(result["A"]) == 1  # 只有 B 的内容
