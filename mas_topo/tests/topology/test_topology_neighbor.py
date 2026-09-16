import numpy as np
from mas_topo.context import ExecutionContext
from mas_topo.topology.neighbor import NeighborTopology


class FakeAgent:
    def __init__(self, name, neighbors):
        self.name = name
        self.neighbors = neighbors


def test_neighbor_topology_from_context_agents():
    agents = [
        FakeAgent("A", [{"name": "B"}, {"name": "C"}]),
        FakeAgent("B", [{"name": "A"}]),
        FakeAgent("C", [{"name": "A"}]),
    ]
    strategy = NeighborTopology()
    context = ExecutionContext()
    context.agents = agents
    adj, rel = strategy.construct_topology(["A", "B", "C"], {}, context)
    expected = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    assert np.array_equal(adj, expected)
    assert rel is None


def test_neighbor_topology_from_explicit_map():
    strategy = NeighborTopology(
        agent_neighbor_map={"A": ["B"], "B": ["A"], "C": []}
    )
    context = ExecutionContext()
    adj, _ = strategy.construct_topology(["A", "B", "C"], {}, context)
    expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert np.array_equal(adj, expected)


def test_neighbor_topology_fallback_to_fully_connected():
    strategy = NeighborTopology()
    context = ExecutionContext()
    adj, _ = strategy.construct_topology(["A", "B", "C"], {}, context)
    expected = np.ones((3, 3), dtype=int)
    np.fill_diagonal(expected, 0)
    assert np.array_equal(adj, expected)
