from mas_topo.message import AgentOutput


def test_private_routing_property():
    out = AgentOutput(metadata={"private_routing": [{"recipients": ["A"], "illocution": "Directive", "content": "x"}]})
    assert out.private_routing == [{"recipients": ["A"], "illocution": "Directive", "content": "x"}]


def test_private_routing_default_empty():
    out = AgentOutput()
    assert out.private_routing == []


def test_vote_complete_property():
    out = AgentOutput(vote_complete=True)
    assert out.vote_complete is True


def test_vote_complete_parsed_from_metadata():
    out = AgentOutput(metadata={"vote_complete": True})
    assert out.vote_complete is True
