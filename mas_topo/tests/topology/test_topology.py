"""拓扑策略模块单元测试。"""

import numpy as np
import pytest
from mas_topo.context import ExecutionContext
from mas_topo.topology.fixed import FixedTopology
from mas_topo.topology.semantic import SemanticMatchingTopology
from mas_topo.topology.cosine import CosineTopology
from mas_topo.topology.manual import ManualTopology


class TestFixedTopology:
    """FixedTopology 预定义模式测试。"""

    def test_full_connected(self):
        topo = FixedTopology(pattern="full_connected")
        adj, rel = topo.construct_topology(["A", "B", "C"], None, ExecutionContext())
        assert adj.shape == (3, 3)
        assert adj[0, 1] == 1 and adj[0, 2] == 1
        assert adj[1, 0] == 1 and adj[2, 0] == 1
        assert np.all(np.diag(adj) == 0)
        assert rel is None

    def test_chain(self):
        topo = FixedTopology(pattern="chain")
        adj, _ = topo.construct_topology(["A", "B", "C", "D"], None, ExecutionContext())
        expected = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ])
        assert np.array_equal(adj, expected)

    def test_star(self):
        topo = FixedTopology(pattern="star")
        adj, _ = topo.construct_topology(["Center", "A", "B", "C"], None, ExecutionContext())
        assert adj[0, 1] == 1 and adj[1, 0] == 1
        assert adj[0, 2] == 1 and adj[2, 0] == 1
        assert adj[0, 3] == 1 and adj[3, 0] == 1
        assert adj[1, 2] == 0

    def test_ring(self):
        topo = FixedTopology(pattern="ring")
        adj, _ = topo.construct_topology(["A", "B", "C", "D"], None, ExecutionContext())
        assert adj[0, 1] == 1 and adj[1, 2] == 1
        assert adj[2, 3] == 1 and adj[3, 0] == 1  # 环闭合
        assert adj[0, 2] == 0  # 非相邻无连接

    def test_tree(self):
        topo = FixedTopology(pattern="tree")
        adj, _ = topo.construct_topology(["A", "B", "C", "D", "E", "F", "G"],
                                         None, ExecutionContext())
        # 二叉树: 0↔1, 0↔2, 1↔3, 1↔4, 2↔5, 2↔6
        assert adj[0, 1] == 1 and adj[1, 0] == 1
        assert adj[0, 2] == 1 and adj[2, 0] == 1
        assert adj[1, 3] == 1 and adj[3, 1] == 1
        assert adj[1, 4] == 1 and adj[4, 1] == 1
        assert adj[2, 5] == 1 and adj[5, 2] == 1
        assert adj[2, 6] == 1 and adj[6, 2] == 1

    def test_mesh(self):
        topo = FixedTopology(pattern="mesh")
        adj, _ = topo.construct_topology(["A", "B", "C", "D"], None, ExecutionContext())
        assert adj[0, 1] == 1 and adj[1, 0] == 1
        assert adj[0, 2] == 1 and adj[2, 0] == 1
        assert adj[1, 3] == 1 and adj[3, 1] == 1
        assert adj[2, 3] == 1 and adj[3, 2] == 1

    def test_layered(self):
        topo = FixedTopology(pattern="layered", seed=42)
        adj, _ = topo.construct_topology(["A", "B", "C", "D"], None, ExecutionContext())
        assert adj.shape == (4, 4)
        assert np.all(np.diag(adj) == 0)

    def test_random(self):
        topo = FixedTopology(pattern="random", seed=42)
        adj, _ = topo.construct_topology(["A", "B", "C"], None, ExecutionContext())
        assert adj.shape == (3, 3)
        assert np.all(np.diag(adj) == 0)

    def test_direct_answer(self):
        topo = FixedTopology(pattern="direct_answer")
        adj, _ = topo.construct_topology(["A"], None, ExecutionContext())
        assert adj.shape == (1, 1)
        assert adj[0, 0] == 0

    def test_invalid_pattern(self):
        with pytest.raises(ValueError, match="Unknown pattern"):
            FixedTopology(pattern="nonexistent")

    def test_empty_agents(self):
        topo = FixedTopology(pattern="chain")
        adj, rel = topo.construct_topology([], None, ExecutionContext())
        assert adj.shape == (0, 0)
        assert rel is None

    def test_reproducible_seed(self):
        topo1 = FixedTopology(pattern="random", seed=123)
        topo2 = FixedTopology(pattern="random", seed=123)
        adj1, _ = topo1.construct_topology(["A", "B", "C", "D", "E"], None, ExecutionContext())
        adj2, _ = topo2.construct_topology(["A", "B", "C", "D", "E"], None, ExecutionContext())
        assert np.array_equal(adj1, adj2)


class TestSemanticMatchingTopology:
    """SemanticMatchingTopology 语义匹配拓扑测试。"""

    def test_empty_agents(self):
        topo = SemanticMatchingTopology(threshold=0.3)
        adj, rel = topo.construct_topology([], None, ExecutionContext())
        assert adj.shape == (0, 0)
        assert rel is None

    def test_no_outputs_fallback(self):
        """首轮无 agent_outputs 时退化为全连接。"""
        topo = SemanticMatchingTopology(threshold=0.3)
        adj, rel = topo.construct_topology(["A", "B", "C"], None, ExecutionContext())
        assert adj.shape == (3, 3)
        assert np.all(np.diag(adj) == 0)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert adj[i, j] == 1

    def test_with_semantic_descriptors(self):
        """使用语义描述符时应生成有意义的邻接矩阵。"""
        from mas_topo.message import AgentOutput

        topo = SemanticMatchingTopology(
            threshold=0.3, max_in_degree=2,
            embedding_model="all-MiniLM-L6-v2")

        outputs = {
            "A": AgentOutput(
                query_descriptor="I need Python code for a palindrome function",
                key_descriptor="I can write Python functions and algorithms",
            ),
            "B": AgentOutput(
                query_descriptor="I need help with code testing and verification",
                key_descriptor="I can test code and find bugs",
            ),
            "C": AgentOutput(
                query_descriptor="I need architecture design patterns",
                key_descriptor="I design system architectures and APIs",
            ),
        }
        adj, rel = topo.construct_topology(["A", "B", "C"], outputs, ExecutionContext())

        assert adj.shape == (3, 3)
        assert np.all(np.diag(adj) == 0)
        assert rel is not None
        assert rel.shape == (3, 3)

    def test_max_in_degree_constraint(self):
        """max_in_degree 约束：每个节点的入度不超过限制。"""
        from mas_topo.message import AgentOutput

        topo = SemanticMatchingTopology(
            threshold=0.001, max_in_degree=1,
            embedding_model="all-MiniLM-L6-v2")

        outputs = {
            name: AgentOutput(
                query_descriptor=f"Agent {name} needs help with task {name}",
                key_descriptor=f"Agent {name} offers expertise in area {name}",
            )
            for name in ["A", "B", "C", "D", "E"]
        }

        adj, _ = topo.construct_topology(list(outputs.keys()), outputs, ExecutionContext())

        in_degrees = adj.sum(axis=0)
        assert all(d <= 1 for d in in_degrees), f"max_in_degree violated: {in_degrees}"

    def test_default_parameters(self):
        topo = SemanticMatchingTopology()
        assert topo.threshold == 0.3
        assert topo.max_in_degree == 3
        assert topo.embedding_model == "all-MiniLM-L6-v2"


class TestCosineTopology:
    """CosineTopology 余弦相似度拓扑测试。"""

    def test_empty_agents(self):
        topo = CosineTopology()
        adj, rel = topo.construct_topology([], None, ExecutionContext())
        assert adj.shape == (0, 0)
        assert rel is None

    def test_full_connected_when_no_outputs(self):
        topo = CosineTopology()
        adj, rel = topo.construct_topology(["A", "B", "C"], None, ExecutionContext())
        assert adj.shape == (3, 3)
        assert np.all(np.diag(adj) == 0)

    def test_with_outputs(self):
        from mas_topo.message import AgentOutput

        topo = CosineTopology(embedding_model="all-MiniLM-L6-v2")
        outputs = {
            "A": AgentOutput(public_content="Python code implementation of sorting algorithms"),
            "B": AgentOutput(public_content="Testing and quality assurance for Python code"),
            "C": AgentOutput(public_content="Mathematical analysis of computational complexity"),
        }
        adj, rel = topo.construct_topology(
            ["A", "B", "C"], outputs, ExecutionContext(task="Write a sorting function"))
        assert adj.shape == (3, 3)
        assert np.all(np.diag(adj) == 0)
        assert rel is not None


class TestManualTopology:
    """ManualTopology 手动邻接矩阵拓扑测试。"""

    def test_explicit_adjacency(self):
        adj_input = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
        topo = ManualTopology(adjacency=adj_input)
        adj, rel = topo.construct_topology(["A", "B", "C"], None, ExecutionContext())
        assert np.array_equal(adj, np.array(adj_input))
        assert rel is None

    def test_empty_agents(self):
        topo = ManualTopology(adjacency=[])
        adj, rel = topo.construct_topology([], None, ExecutionContext())
        assert adj.shape == (0, 0)

    def test_validation_mismatch(self):
        topo = ManualTopology(adjacency=[[0, 1], [0, 0]])
        with pytest.raises(ValueError):
            topo.construct_topology(["A", "B", "C"], None, ExecutionContext())
