import json
from pathlib import Path

import pytest

from visualization.visualize_server import create_app


@pytest.fixture
def client(tmp_path: Path):
    exp_dir = tmp_path / "experiment_results"
    (exp_dir / "run_a" / "trace" / "sample_000").mkdir(parents=True)
    (exp_dir / "run_a" / "trace" / "sample_000" / "trace.json").write_text(
        json.dumps({
            "rounds": [
                {"round_num": 1, "edges": [{"source": "A", "target": "B", "content": "hello"}]},
                {"round_num": 2, "edges": [{"source": "B", "target": "C", "content": "world"}]},
            ]
        })
    )
    app = create_app(str(exp_dir))
    app.config["TESTING"] = True
    return app.test_client()


def test_index_page(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"PragmaMAS Visualizer" in resp.data


def test_api_runs(client):
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "run_a" in data["runs"]


def test_api_samples(client):
    resp = client.get("/api/runs/run_a/samples")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "sample_000" in data["samples"]


def test_api_trace(client):
    resp = client.get("/api/runs/run_a/samples/sample_000")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "trace" in data
    assert "round_tree" in data
    assert len(data["trace"]["rounds"]) == 2


def test_api_trace_not_found(client):
    resp = client.get("/api/runs/run_a/samples/sample_999")
    assert resp.status_code == 404
