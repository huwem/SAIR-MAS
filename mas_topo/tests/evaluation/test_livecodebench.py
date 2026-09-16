import base64
import json
import pickle
import zlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from mas_topo.datasets.livecodebench_loader import LiveCodeBenchLoader
from mas_topo.evaluation.code_execution_stdin import CodeExecutionStdinEvaluator


PUBLIC = [{"input": "1", "output": "1"}]
PRIVATE = [{"input": "2", "output": "2"}]


def encode_private(cases, encoding):
    text = json.dumps(cases)
    if encoding == "list":
        return cases
    if encoding == "json":
        return text
    payload = pickle.dumps(text) if encoding == "pickle" else text.encode()
    return base64.b64encode(zlib.compress(payload)).decode()


def load_sample(tmp_path, private):
    path = tmp_path / "lcb.jsonl"
    path.write_text(json.dumps({
        "question_id": "echo", "question_content": "Echo the input.",
        "public_test_cases": json.dumps(PUBLIC), "private_test_cases": private,
    }) + "\n")
    return LiveCodeBenchLoader(str(path)).load()[0]


@pytest.mark.parametrize("encoding", ["list", "json", "compressed", "pickle"])
def test_public_only_solution_fails_full_suite(tmp_path, encoding):
    sample = load_sample(tmp_path, encode_private(PRIVATE, encoding))
    result = CodeExecutionStdinEvaluator().evaluate(sample, "print(1)")
    assert result.is_correct is False
    assert result.metric_value == 0.0
    assert result.details["suite"] == "public+private"
    assert result.details["n_public_tests"] == 1
    assert result.details["n_private_tests"] == 1
    assert result.details["n_tests"] == 2
    assert result.details["n_passed"] == 1
    assert result.details["test_pass_rate"] == 0.5
    assert len(result.details["cases"]) == 2
    # ground_truth is injected into graph metadata, so it must remain public-only.
    assert json.loads(sample.ground_truth) == PUBLIC
    assert sample.public_tests == PUBLIC
    assert sample.private_tests == PRIVATE


def test_correct_solution_passes_full_suite(tmp_path):
    sample = load_sample(tmp_path, PRIVATE)
    result = CodeExecutionStdinEvaluator().evaluate(sample, "print(input())")
    assert result.is_correct is True
    assert result.metric_value == 1.0
    assert result.details["n_passed"] == result.details["n_tests"] == 2


@pytest.mark.parametrize("prediction", ["", "not python"])
def test_invalid_answer_retains_suite_provenance(tmp_path, prediction):
    sample = load_sample(tmp_path, PRIVATE)
    result = CodeExecutionStdinEvaluator().evaluate(sample, prediction)
    assert result.is_correct is False
    assert result.metric_value == 0.0
    assert result.details["suite"] == "public+private"
    assert result.details["n_tests"] == 2
    assert result.details["n_executed"] == 0


@pytest.mark.parametrize("private", ["corrupted-base64", {"unexpected": "object"},
                                      ["not a case"], [{"input": "2"}], None, ""])
def test_invalid_private_suite_fails_at_load(tmp_path, private):
    with pytest.raises(ValueError, match="private_test_cases"):
        load_sample(tmp_path, private)


def test_missing_private_suite_fails_at_load(tmp_path):
    path = tmp_path / "missing.jsonl"
    path.write_text(json.dumps({"public_test_cases": PUBLIC}) + "\n")
    with pytest.raises(ValueError, match="private_test_cases"):
        LiveCodeBenchLoader(str(path)).load()


def test_generic_ground_truth_fallback_still_runs():
    sample = SimpleNamespace(ground_truth=json.dumps(PUBLIC + PRIVATE))
    result = CodeExecutionStdinEvaluator().evaluate(sample, "print(input())")
    assert result.is_correct is True
    assert result.details["suite"] == "ground_truth"
    assert result.details["n_tests"] == 2


def test_explicit_empty_suite_cannot_pass(tmp_path):
    sample = load_sample(tmp_path, [])
    sample.public_tests = []
    result = CodeExecutionStdinEvaluator().evaluate(sample, "print(1)")
    assert result.is_correct is False
    assert result.metric_value == 0.0
    assert result.details["n_tests"] == 0


def test_runner_exports_full_suite_without_exposing_private_tests(tmp_path, monkeypatch):
    from mas_topo.config.schema import FrameworkConfig
    from mas_topo.context import ExecutionContext
    from mas_topo.experiment import dataset_runner
    from mas_topo.graph import GraphResult

    load_sample(tmp_path, encode_private(PRIVATE, "pickle"))
    graph = MagicMock()
    graph.run.return_value = GraphResult(
        final_answer="print(1)", rounds=[],
        context=ExecutionContext(task="Echo the input."), is_completed=True,
    )
    monkeypatch.setattr(dataset_runner.Graph, "from_config", lambda *a, **kw: graph)
    output = tmp_path / "results"
    runner = dataset_runner.DatasetExperimentRunner(
        config=FrameworkConfig(), evaluator=CodeExecutionStdinEvaluator(),
        output_dir=str(output), verbose=False, progress_bar=False,
    )
    summary = runner.run_on_dataset(LiveCodeBenchLoader(str(tmp_path / "lcb.jsonl")))
    assert summary.results[0].error is None
    assert summary.correct_count == 0
    assert summary.avg_metric == 0.0
    graph.run.assert_called_once_with("Echo the input.", metadata={"test_cases": json.dumps(PUBLIC)})
    saved = json.loads((output / "results.json").read_text())
    assert saved["results"][0]["evaluation"]["details"]["n_private_tests"] == 1
    assert saved["results"][0]["evaluation"]["details"]["suite"] == "public+private"
