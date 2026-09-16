"""DatasetExperimentRunner 样本级墙钟超时测试。

回归：sair_reasoning_local aime 20260903_083025 事故——单样本挂起
（run_python 失控子进程）导致 as_completed 永久阻塞，进度条停在 89/90，
无日志无重试。样本级超时保证挂起样本被记为 error 后全局继续。
"""

import time
from typing import Optional

import numpy as np
import pytest

import mas_topo.experiment.dataset_runner as dataset_runner_module
from mas_topo.config.schema import FrameworkConfig
from mas_topo.context import ExecutionContext
from mas_topo.datasets.base import BaseDatasetLoader, DatasetSample
from mas_topo.evaluation.base import EvaluationResult
from mas_topo.experiment.dataset_runner import DatasetExperimentRunner
from mas_topo.graph import GraphResult, RoundResult
from mas_topo.decision.base import DecisionResult


class _FakeSample(DatasetSample):
    def __init__(self, question: str, ground_truth: str = "42"):
        self._question = question
        self._ground_truth = ground_truth

    @property
    def question(self) -> str:
        return self._question

    @property
    def ground_truth(self) -> str:
        return self._ground_truth

    def to_dict(self) -> dict:
        return {"question": self._question, "ground_truth": self._ground_truth}


class _FakeLoader(BaseDatasetLoader):
    def __init__(self, n: int):
        super().__init__(data_path="fake")
        self._n = n

    def load(self, limit: Optional[int] = None):
        samples = [_FakeSample(f"q{i}") for i in range(self._n)]
        return samples[:limit] if limit is not None else samples

    def total_count(self) -> int:
        return self._n


class _FakeEvaluator:
    metric_name = "exact_match"

    def evaluate(self, sample, answer) -> EvaluationResult:
        return EvaluationResult(
            is_correct=True,
            predicted_answer=str(answer),
            metric_value=1.0,
            metric_name=self.metric_name,
        )


def _fast_graph_result() -> GraphResult:
    r1 = RoundResult(
        round_num=1,
        adjacency_matrix=np.ones((1, 1), dtype=int),
        routed_messages={},
        decision=DecisionResult(is_complete=True, final_answer="42"),
    )
    return GraphResult(
        final_answer="42",
        rounds=[r1],
        context=ExecutionContext(task="t", prompt_tokens=1, completion_tokens=1),
        is_completed=True,
    )


class _FakeGraph:
    """run() 行为可按样本内容区分：问题含 'slow' 时长时间挂起。"""

    hang_seconds = 3.0

    @classmethod
    def from_config(cls, config, topology_logger=None):
        return cls()

    def run(self, task, metadata=None):
        if "slow" in task:
            time.sleep(self.hang_seconds)
        return _fast_graph_result()


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_runner_module, "Graph", _FakeGraph)
    return DatasetExperimentRunner(
        config=FrameworkConfig(),
        evaluator=_FakeEvaluator(),
        verbose=False,
        output_dir=str(tmp_path),
        progress_bar=False,
    )


class TestSampleTimeout:
    """样本级墙钟超时（sample_timeout）。"""

    def test_sequential_hung_sample_times_out(self, runner):
        class _SlowLoader(_FakeLoader):
            def load(self, limit=None):
                return [_FakeSample("slow q")]

        t0 = time.monotonic()
        summary = runner.run_on_dataset(_SlowLoader(1), sample_timeout=0.3)
        elapsed = time.monotonic() - t0

        assert elapsed < _FakeGraph.hang_seconds
        assert len(summary.results) == 1
        assert summary.completed_count == 0
        assert "timed out" in (summary.results[0].error or "")

    def test_batched_hung_sample_does_not_block_others(self, runner):
        class _MixedLoader(_FakeLoader):
            def load(self, limit=None):
                return [_FakeSample("fast q"), _FakeSample("slow q")]

        t0 = time.monotonic()
        summary = runner.run_on_dataset(_MixedLoader(2), batch_size=2, sample_timeout=0.3)
        elapsed = time.monotonic() - t0

        assert elapsed < _FakeGraph.hang_seconds
        assert len(summary.results) == 2
        by_index = {r.index: r for r in summary.results}
        assert by_index[0].is_completed is True
        assert by_index[0].error is None
        assert "timed out" in (by_index[1].error or "")
        assert summary.completed_count == 1

    def test_fast_sample_unaffected_without_timeout(self, runner):
        summary = runner.run_on_dataset(_FakeLoader(1))
        assert summary.completed_count == 1
        assert summary.results[0].error is None
        assert summary.correct_count == 1


class TestSampleTimeoutConfig:
    """sample_timeout 配置项。"""

    def test_dataset_config_default_disabled(self):
        from mas_topo.config.experiment_schema import DatasetConfig

        assert DatasetConfig().sample_timeout == 0.0
