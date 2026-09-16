import json
import subprocess
import sys
from pathlib import Path

import pytest

import visualization.visualize_rounds as viz


def test_plot_topology(tmp_path: Path):
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    round_data = {
        "round_num": 1,
        "topology_strategy": "NeighborTopology",
        "adjacency_list": {"A": ["B", "C"], "B": ["A"], "C": ["A", "B"]},
    }
    viz.plot_topology(round_data, output_dir, "png", "spring")
    assert (output_dir / "topology_round_001.png").exists()


def test_plot_communication(tmp_path: Path):
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    round_data = {
        "round_num": 1,
        "routing_strategy": "SelfRouting",
        "agents": {"A": {}, "B": {}},
        "edges": [
            {"source": "A", "target": "B", "channel": "ft-private",
             "metadata": {"illocution_category": "Directive"}},
        ],
    }
    viz.plot_communication(round_data, output_dir, "png", "spring")
    assert (output_dir / "communication_round_001.png").exists()


def test_plot_metrics(tmp_path: Path):
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    round_data = {
        "round_num": 1,
        "agent_count": 3,
        "edge_count": 2,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "elapsed_time": 1.5,
        "topology_metadata": {"density": 0.25},
    }
    viz.plot_metrics(round_data, output_dir, "png", "spring")
    assert (output_dir / "metrics_round_001.png").exists()


def test_plot_activity(tmp_path: Path):
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    round_data = {
        "round_num": 1,
        "agents": {"A": {}, "B": {}},
        "edges": [
            {"source": "A", "target": "B", "metadata": {"illocution_category": "Directive"}},
            {"source": "B", "target": "A", "metadata": {"illocution_category": "Assertive"}},
            {"source": "A", "target": "B", "metadata": {"illocution_category": "Directive"}},
        ],
    }
    viz.plot_activity(round_data, output_dir, "png", "spring")
    assert (output_dir / "activity_round_001.png").exists()


def test_parse_plots():
    assert viz.parse_plots("all") == ["topology", "communication", "metrics", "activity"]
    assert viz.parse_plots("topology, metrics") == ["topology", "metrics"]
    with pytest.raises(ValueError, match="Invalid plot name"):
        viz.parse_plots("topology,foo")


def test_parse_rounds():
    assert viz.parse_rounds("all", [1, 2, 3]) == [1, 2, 3]
    assert viz.parse_rounds("1,3", [1, 2, 3]) == [1, 3]
    with pytest.raises(ValueError, match="Round 5 not found"):
        viz.parse_rounds("5", [1, 2, 3])


def test_cli_defaults(tmp_path: Path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({
        "rounds": [
            {"round_num": 1, "adjacency_list": {"A": ["B"]}, "agents": {"A": {}, "B": {}}, "edges": []},
        ]
    }))
    result = subprocess.run(
        [sys.executable, "visualization/visualize_rounds.py", str(trace)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figures" / "topology_round_001.png").exists()


def test_cli_plots_selection(tmp_path: Path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({
        "rounds": [
            {"round_num": 1, "adjacency_list": {"A": ["B"]}, "agents": {"A": {}, "B": {}}, "edges": []},
        ]
    }))
    result = subprocess.run(
        [sys.executable, "visualization/visualize_rounds.py", str(trace), "--plots", "topology,metrics"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figures" / "topology_round_001.png").exists()
    assert (tmp_path / "figures" / "metrics_round_001.png").exists()
    assert not (tmp_path / "figures" / "communication_round_001.png").exists()


def test_cli_rounds_selection(tmp_path: Path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({
        "rounds": [
            {"round_num": 1, "adjacency_list": {"A": ["B"]}},
            {"round_num": 2, "adjacency_list": {"A": ["B"]}},
        ]
    }))
    result = subprocess.run(
        [sys.executable, "visualization/visualize_rounds.py", str(trace), "--rounds", "2", "--plots", "topology"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "figures" / "topology_round_001.png").exists()
    assert (tmp_path / "figures" / "topology_round_002.png").exists()


def test_cli_format_pdf(tmp_path: Path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({
        "rounds": [{"round_num": 1, "adjacency_list": {"A": ["B"]}}]
    }))
    result = subprocess.run(
        [sys.executable, "visualization/visualize_rounds.py", str(trace), "--format", "pdf", "--plots", "topology"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figures" / "topology_round_001.pdf").exists()


def test_cli_invalid_plot_name(tmp_path: Path):
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"rounds": [{"round_num": 1}]}))
    result = subprocess.run(
        [sys.executable, "visualization/visualize_rounds.py", str(trace), "--plots", "foo"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Invalid plot name" in result.stderr
