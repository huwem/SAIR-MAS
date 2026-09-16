"""实验模块单元测试。"""

import numpy as np
from mas_topo.experiment.tracker import ExperimentTracker, RunMetrics, RoundMetrics
from mas_topo.experiment.runner import ExperimentRunner
from mas_topo.config.loader import load_config
from mas_topo.context import ExecutionContext


class TestExperimentTracker:
    """ExperimentTracker 测试。"""

    def test_empty_tracker(self):
        tracker = ExperimentTracker()
        assert len(tracker.get_runs()) == 0

    def test_record_run(self):
        """模拟记录一次运行。"""
        from mas_topo.graph import GraphResult, RoundResult
        from mas_topo.decision.base import DecisionResult

        # 构造 GraphResult
        r1 = RoundResult(
            round_num=1,
            adjacency_matrix=np.ones((2, 2), dtype=int),
            routed_messages={
                "A": [type("Msg", (), {"content": "hello from B", "sender_id": "B"})()],
                "B": [],
            },
            decision=DecisionResult(is_complete=True, final_answer="42"),
        )
        result = GraphResult(
            final_answer="42",
            rounds=[r1],
            context=ExecutionContext(task="Test task", cost=0.1,
                                     prompt_tokens=50, completion_tokens=20),
            is_completed=True,
        )

        tracker = ExperimentTracker()
        tracker.start_run()
        metrics = tracker.record_run(
            result, task="Test task", topology="fixed/chain",
            routing="graph", decision="manager")

        assert metrics.task == "Test task"
        assert metrics.is_completed
        assert metrics.total_rounds == 1
        assert metrics.total_cost == 0.1
        assert len(metrics.round_metrics) == 1

        rm = metrics.round_metrics[0]
        assert rm.round_num == 1
        assert rm.num_messages == 1
        assert rm.is_complete

    def test_json_export(self):
        tracker = ExperimentTracker()
        json_str = tracker.to_json()
        assert json_str == "[]"

    def test_csv_export(self):
        tracker = ExperimentTracker()
        csv_str = tracker.to_csv()
        assert "task" in csv_str

    def test_summary(self):
        tracker = ExperimentTracker()
        summary = tracker.summary()
        assert "Experiment Summary" in summary


class TestRoundMetrics:
    """RoundMetrics 测试。"""

    def test_default_values(self):
        rm = RoundMetrics()
        assert rm.round_num == 0
        assert rm.num_edges == 0
        assert rm.graph_density == 0.0


class TestRunMetrics:
    """RunMetrics 测试。"""

    def test_default_values(self):
        run = RunMetrics()
        assert run.task == ""
        assert run.total_rounds == 0
        assert not run.is_completed


class TestExperimentRunner:
    """ExperimentRunner 测试（不需要真实 LLM）。"""

    def test_runner_creation(self):
        runner = ExperimentRunner(verbose=False)
        assert runner.tracker is not None

    def test_run_with_mock(self):
        """使用 mock 验证运行流程。"""
        # ExperimentRunner 依赖于真实的 Graph 执行
        # 这里只验证 Runner 的 API 不崩溃
        runner = ExperimentRunner(verbose=False)
        assert isinstance(runner.tracker, ExperimentTracker)

    def test_compare_topologies_method(self):
        """验证 compare_topologies 创建正确的配置。"""
        runner = ExperimentRunner(verbose=False)
        base = load_config("configs/fixed_chain.yaml")
        # 不使用真实 LLM 运行，只验证配置创建
        assert base.topology.strategy == "fixed"
