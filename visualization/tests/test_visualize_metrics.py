import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import visualization.visualize_metrics as viz



def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_resolve_paths_from_experiment_dir(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [{"sample_index": 0, "total_rounds": 1}])
    mp, od = viz.resolve_paths(str(tmp_path), None, None)
    assert Path(mp).name == "metrics.jsonl"
    assert od == str(tmp_path / "figures")


def test_resolve_paths_prefers_jsonl_over_csv(tmp_path: Path):
    (tmp_path / "metrics.jsonl").write_text('{"sample_index": 0}\n')
    (tmp_path / "metrics.csv").write_text("sample_index\n0\n")
    mp, _ = viz.resolve_paths(str(tmp_path), None, None)
    assert Path(mp).name == "metrics.jsonl"


def test_resolve_paths_custom_output_dir(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [{"sample_index": 0}])
    _, od = viz.resolve_paths(str(tmp_path), None, str(tmp_path / "custom_figures"))
    assert od == str(tmp_path / "custom_figures")


def test_resolve_paths_metrics_requires_output_dir(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [{"sample_index": 0}])
    with pytest.raises(ValueError, match="--output-dir"):
        viz.resolve_paths(None, str(metrics), None)


def test_resolve_paths_missing_experiment_dir(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        viz.resolve_paths(str(missing), None, None)


def test_parse_plots_all():
    assert viz.parse_plots("all") == [
        "accuracy", "rounds", "tokens", "topology", "activation", "illocution", "channel"
    ]


def test_parse_plots_subset():
    assert viz.parse_plots("accuracy, tokens ") == ["accuracy", "tokens"]


def test_parse_plots_invalid():
    with pytest.raises(ValueError, match="Invalid plot name"):
        viz.parse_plots("accuracy,foo")


def test_load_metrics_empty_jsonl(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("")
    df = viz.load_metrics(str(metrics))
    assert df.empty


def test_load_metrics_minimal_run(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        {"sample_index": 0, "total_rounds": 2, "is_completed": True, "is_correct": True,
         "rounds": [{"round_num": 1, "num_messages": 3}, {"round_num": 2, "num_messages": 4}]}
    ])
    df = viz.load_metrics(str(metrics))
    assert len(df) == 2
    assert df["round_num"].tolist() == [1, 2]


def test_print_summary(capsys: pytest.CaptureFixture):
    df = pd.DataFrame([
        {"sample_index": 0, "is_completed": True, "is_correct": True, "total_rounds": 3,
         "total_prompt_tokens": 1000, "total_completion_tokens": 200},
        {"sample_index": 1, "is_completed": False, "is_correct": False, "total_rounds": 5,
         "total_prompt_tokens": 2000, "total_completion_tokens": 400},
    ])
    viz.print_summary(df)
    out = capsys.readouterr().out
    assert "Samples:                 2" in out
    assert "Completed:               1 (50.0%)" in out
    assert "Correct:                 1 (50.0%)" in out
    assert "Avg rounds:              4.00" in out
    assert "Avg prompt tokens:       1,500" in out
    assert "Avg completion tokens:   300" in out


def test_print_summary_empty():
    df = pd.DataFrame()
    viz.print_summary(df)  # should not raise


def test_cli_experiment_dir_defaults(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        {"sample_index": 0, "total_rounds": 1, "is_completed": True, "is_correct": True,
         "total_prompt_tokens": 100, "total_completion_tokens": 50}
    ])
    result = subprocess.run(
        [sys.executable, "visualization/visualize_metrics.py", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    figures_dir = tmp_path / "figures"
    assert (figures_dir / "accuracy_overview.png").exists()
    assert (figures_dir / "rounds_distribution.png").exists()
    assert "Summary" in result.stdout


def test_cli_plots_selection(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        {"sample_index": 0, "total_rounds": 1, "is_completed": True, "is_correct": True}
    ])
    result = subprocess.run(
        [sys.executable, "visualization/visualize_metrics.py", str(tmp_path), "--plots", "accuracy,rounds"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figures" / "accuracy_overview.png").exists()
    assert (tmp_path / "figures" / "rounds_distribution.png").exists()
    assert not list((tmp_path / "figures").glob("token_usage.*"))


def test_cli_format_pdf(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        {"sample_index": 0, "total_rounds": 1, "is_completed": True, "is_correct": True}
    ])
    result = subprocess.run(
        [sys.executable, "visualization/visualize_metrics.py", str(tmp_path),
         "--format", "pdf", "--plots", "accuracy"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figures" / "accuracy_overview.pdf").exists()


def test_cli_empty_data(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("")
    result = subprocess.run(
        [sys.executable, "visualization/visualize_metrics.py", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "No metrics data" in result.stdout


def test_cli_invalid_plot_name(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        {"sample_index": 0, "total_rounds": 1, "is_completed": True, "is_correct": True}
    ])
    result = subprocess.run(
        [sys.executable, "visualization/visualize_metrics.py", str(tmp_path), "--plots", "accuracy,foo"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "Invalid plot name" in result.stderr


def test_filter_samples_defaults(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        {"sample_index": 2, "total_rounds": 1},
        {"sample_index": 0, "total_rounds": 1},
        {"sample_index": 1, "total_rounds": 1},
    ])
    df = viz.load_metrics(str(metrics))
    filtered = viz.filter_samples(df, 0, -1)
    assert sorted(filtered["sample_index"].unique().tolist()) == [0, 1, 2]


def test_filter_samples_subset(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        {"sample_index": 0, "total_rounds": 1},
        {"sample_index": 1, "total_rounds": 1},
        {"sample_index": 2, "total_rounds": 1},
        {"sample_index": 3, "total_rounds": 1},
    ])
    df = viz.load_metrics(str(metrics))
    filtered = viz.filter_samples(df, 1, 2)
    assert filtered["sample_index"].unique().tolist() == [1, 2]


def test_filter_samples_clamps(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        {"sample_index": 0, "total_rounds": 1},
        {"sample_index": 1, "total_rounds": 1},
    ])
    df = viz.load_metrics(str(metrics))
    filtered = viz.filter_samples(df, 0, 5)
    assert filtered["sample_index"].unique().tolist() == [0, 1]


def test_filter_samples_offset_exceeds(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [{"sample_index": 0, "total_rounds": 1}])
    df = viz.load_metrics(str(metrics))
    with pytest.raises(ValueError, match="exceeds"):
        viz.filter_samples(df, 5, 1)


def test_filter_samples_negative_num_invalid(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [{"sample_index": 0, "total_rounds": 1}])
    df = viz.load_metrics(str(metrics))
    with pytest.raises(ValueError, match="--num"):
        viz.filter_samples(df, 0, -2)


def test_print_summary_with_total(capsys: pytest.CaptureFixture):
    df = pd.DataFrame([
        {"sample_index": 0, "is_completed": True, "is_correct": True, "total_rounds": 3,
         "total_prompt_tokens": 1000, "total_completion_tokens": 200},
    ])
    viz.print_summary(df, total_samples=10)
    out = capsys.readouterr().out
    assert "Samples:                 1 / 10" in out


def test_cli_offset_num(tmp_path: Path):
    metrics = tmp_path / "metrics.jsonl"
    _write_jsonl(metrics, [
        {"sample_index": i, "total_rounds": 1, "is_completed": True, "is_correct": True}
        for i in range(5)
    ])
    result = subprocess.run(
        [sys.executable, "visualization/visualize_metrics.py", str(tmp_path),
         "--offset", "1", "--num", "2", "--plots", "accuracy"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Samples:                 2 / 5" in result.stdout
    assert (tmp_path / "figures" / "accuracy_overview.png").exists()
