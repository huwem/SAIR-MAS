"""数据集批量实验运行器。

支持从配置文件加载、在数据集上批量运行实验、
自动评估准确率、导出结果到 JSON/CSV。

过程追踪（每样本每轮 Agent I/O、拓扑、路由决策、计时）与
实验结果（准确率、答案）严格分离到不同文件。
"""

import json
import os
import threading
import time
import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from mas_topo.config.schema import FrameworkConfig
from mas_topo.datasets.base import BaseDatasetLoader, DatasetSample
from mas_topo.evaluation.base import Evaluator, EvaluationResult
from mas_topo.graph import Graph
from mas_topo.experiment.tracker import ExperimentTracker
from mas_topo.experiment.topology_logger import write_results_file

logger = logging.getLogger(__name__)


@dataclass
class DatasetRunResult:
    """单个数据集样本的运行结果。"""

    index: int = 0
    question: str = ""
    ground_truth: str = ""
    predicted_answer: str = ""
    is_correct: bool = False
    is_completed: bool = False
    total_rounds: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_time: float = 0.0
    final_answer: str = ""
    error: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None


@dataclass
class DatasetExperimentSummary:
    """数据集实验汇总结果。"""

    dataset_path: str = ""
    total_samples: int = 0
    completed_count: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    metric_name: str = ""
    avg_metric: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_elapsed_time: float = 0.0
    avg_rounds: float = 0.0
    results: list[DatasetRunResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "total_samples": self.total_samples,
            "completed_count": self.completed_count,
            "correct_count": self.correct_count,
            "accuracy": round(self.accuracy, 4),
            "metric_name": self.metric_name,
            "avg_metric": round(self.avg_metric, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_elapsed_time": round(self.total_elapsed_time, 2),
            "avg_rounds": round(self.avg_rounds, 2),
            "results": [
                {
                    "index": r.index,
                    "question": r.question,
                    "ground_truth": r.ground_truth,
                    "predicted_answer": r.predicted_answer,
                    "is_correct": r.is_correct,
                    "is_completed": r.is_completed,
                    "total_rounds": r.total_rounds,
                    "prompt_tokens": r.prompt_tokens,
                    "completion_tokens": r.completion_tokens,
                    "elapsed_time": round(r.elapsed_time, 2),
                    "final_answer": r.final_answer,
                    "error": r.error,
                    "evaluation": r.evaluation.to_dict() if r.evaluation else None,
                }
                for r in self.results
            ],
        }


class DatasetExperimentRunner:
    """数据集批量实验运行器。

    通过配置文件驱动，在数据集上批量运行多智能体实验，
    自动评估准确率并导出结果。

    文件输出（当 config.log_dir 设置时）:
        {log_dir}/
        ├── trace/                 # 过程追踪
        │   ├── sample_000/
        │   │   ├── trace.json
        │   │   ├── round_*.md
        │   │   └── index.md
        │   ├── sample_001/
        │   │   └── ...
        │   └── index.md
        ├── results.json           # 实验结果（准确率、答案等）
        ├── results.csv            # CSV 格式结果
        ├── metrics.jsonl          # 每样本每轮指标（JSON Lines，便于增量采集）
        ├── metrics.json           # 聚合指标（JSON）
        └── metrics.csv            # 聚合指标（CSV，每行一轮）
    """

    def __init__(
        self,
        config: FrameworkConfig,
        evaluator: Evaluator,
        verbose: bool = True,
        output_dir: Optional[str] = None,
        progress_bar: bool = True,
    ):
        self.config = config
        self.evaluator = evaluator
        self.verbose = verbose
        self.progress_bar = progress_bar
        self.output_dir = Path(output_dir) if output_dir else Path("experiment_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 确定日志目录
        if config.log_dir:
            self.log_dir = Path(config.log_dir)
        else:
            self.log_dir = self.output_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 全局指标跟踪器与线程锁
        self.tracker = ExperimentTracker()
        self._tracker_lock = threading.Lock()
        self._save_lock = threading.Lock()

    def run_on_dataset(
        self,
        dataset_loader: BaseDatasetLoader,
        limit: Optional[int] = None,
        offset: int = 0,
        batch_size: int = 1,
        sample_timeout: float = 0.0,
    ) -> DatasetExperimentSummary:
        """在数据集上运行批量实验。

        Args:
            dataset_loader: 数据集加载器（如 GSM8KLoader），需实现 load() 方法。
            limit: 最多运行的样本数。
            offset: 从第几个样本开始运行。
            batch_size: 并发执行的样本数（默认 1，即串行）。设为 >1 时使用线程池并发执行。
            sample_timeout: 单样本墙钟超时（秒），<=0 表示不限制。超时的样本记为
                error 并继续后续样本；其后台线程可能继续运行直到自然结束。

        Returns:
            DatasetExperimentSummary 汇总结果。
        """
        load_limit = (limit + offset) if limit is not None else None
        samples = dataset_loader.load(limit=load_limit)
        if offset > 0:
            samples = samples[offset:]
        if limit is not None:
            samples = samples[:limit]

        summary = DatasetExperimentSummary(
            dataset_path=str(getattr(dataset_loader, "data_path", "")),
            total_samples=len(samples),
            metric_name=getattr(self.evaluator, "metric_name", "") or "",
        )

        if batch_size > 1:
            self._run_batched(samples, summary, offset, batch_size, sample_timeout)
        else:
            self._run_sequential(samples, summary, offset, sample_timeout)

        return summary

    def _run_sequential(
        self, samples: list[DatasetSample], summary: DatasetExperimentSummary,
        offset: int, sample_timeout: float = 0.0,
    ) -> None:
        """串行执行样本。"""
        total = len(samples)
        pbar = tqdm(total=total, desc="Running", unit="sample", disable=not self.progress_bar)
        for idx, sample in enumerate(samples):
            global_idx = offset + idx
            if self.verbose and not self.progress_bar:
                self._print_sample_header(sample, global_idx, offset, total)
            result = self._run_single_with_timeout(sample, global_idx, sample_timeout)
            self._accumulate_result(summary, result)
            self._update_and_save(summary)
            self._log_sample_result(result)
            pbar.set_postfix_str(
                f"pass {summary.correct_count}/{summary.completed_count}/{len(summary.results)}",
                refresh=False,
            )
            pbar.update(1)
        pbar.close()

    def _run_single_with_timeout(
        self, sample: DatasetSample, index: int, sample_timeout: float
    ) -> DatasetRunResult:
        """带墙钟超时地运行单个样本（超时则在守护线程中放弃等待）。"""
        if not sample_timeout or sample_timeout <= 0:
            return self._run_single(sample, index)

        box: dict = {}
        t_start = time.monotonic()
        t = threading.Thread(
            target=lambda: box.setdefault("result", self._run_single(sample, index)),
            daemon=True,
        )
        t.start()
        t.join(sample_timeout)
        if not t.is_alive():
            return box["result"]
        return self._timeout_result(sample, index, t_start, sample_timeout)

    def _timeout_result(
        self, sample: DatasetSample, index: int, t_start: float, sample_timeout: float
    ) -> DatasetRunResult:
        """为挂起样本构造超时结果（is_completed=False，error 记录原因）。"""
        logger.warning(
            "Sample %d timed out after %.0fs; abandoning "
            "(worker thread may still finish in background)",
            index, sample_timeout,
        )
        result = DatasetRunResult(
            index=index,
            question=sample.question,
            ground_truth=sample.ground_truth,
        )
        result.error = f"Sample timed out after {sample_timeout:.0f}s (wall-clock)"
        result.elapsed_time = time.monotonic() - t_start
        return result

    def _run_batched(
        self, samples: list[DatasetSample], summary: DatasetExperimentSummary,
        offset: int, batch_size: int, sample_timeout: float = 0.0,
    ) -> None:
        """并发执行样本。

        用 wait(FIRST_COMPLETED) 轮询替代 as_completed：挂起样本的 future
        永不完成时，as_completed 会永久阻塞主线程（曾致实验停在 89/90）；
        轮询循环在样本超过 sample_timeout 后记为超时并继续。
        """
        total = len(samples)
        pbar = tqdm(total=total, desc="Running", unit="sample", disable=not self.progress_bar)
        if self.verbose:
            pbar.write(f"Running {total} samples with batch_size={batch_size}")
        executor = ThreadPoolExecutor(max_workers=batch_size)
        timed_out: list[int] = []

        def _record(result: DatasetRunResult) -> None:
            self._accumulate_result(summary, result)
            self._update_and_save(summary)
            self._log_sample_result(result)
            pbar.set_postfix_str(
                f"pass {summary.correct_count}/{summary.completed_count}/{len(summary.results)}",
                refresh=False,
            )
            pbar.update(1)

        try:
            pending: dict = {
                executor.submit(self._run_single, sample, offset + idx): (
                    offset + idx, sample, time.monotonic()
                )
                for idx, sample in enumerate(samples)
            }
            while pending:
                # 轮询间隔不超过最近一个样本的剩余时限，保证超时按时触发
                poll = 1.0
                if sample_timeout and sample_timeout > 0:
                    now = time.monotonic()
                    remaining = min(
                        t0 + sample_timeout - now for _, _, t0 in pending.values()
                    )
                    poll = max(0.05, min(1.0, remaining))
                done, _ = wait(list(pending), timeout=poll, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    _record(future.result())
                if sample_timeout and sample_timeout > 0:
                    now = time.monotonic()
                    for future, (idx, sample, t0) in list(pending.items()):
                        if now - t0 >= sample_timeout:
                            pending.pop(future)
                            future.cancel()  # 仅对尚未开始的 future 有效
                            timed_out.append(idx)
                            _record(self._timeout_result(sample, idx, t0, sample_timeout))
        except KeyboardInterrupt:
            pbar.write("[Interrupted] Cancelling pending samples and saving partial results...")
            executor.shutdown(wait=False, cancel_futures=True)
            pbar.close()
            raise
        if timed_out:
            logger.warning(
                "%d sample(s) timed out and were abandoned: %s",
                len(timed_out), timed_out,
            )
            # 泄漏的工作线程可能仍在跑，不等它们结束（进程退出前会自然收尾）
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)
        pbar.close()

    def _accumulate_result(self, summary: DatasetExperimentSummary, result: DatasetRunResult) -> None:
        """将单个结果累积到汇总中。"""
        summary.results.append(result)
        if result.is_correct:
            summary.correct_count += 1
        if result.is_completed:
            summary.completed_count += 1
        summary.total_prompt_tokens += result.prompt_tokens
        summary.total_completion_tokens += result.completion_tokens
        summary.total_elapsed_time += result.elapsed_time

    def _print_sample_header(self, sample: DatasetSample, global_idx: int, offset: int, total: int) -> None:
        """仅在无进度条模式下打印样本头（否则靠 logger 记录）。"""
        if not self.verbose:
            return
        print(f"\n[{'='*60}]")
        print(f"[{global_idx+1}/{offset+total}] Question:")
        print(f"  {sample.question[:100]}...")
        print(f"[{'='*60}]")

    def _log_sample_result(self, result: DatasetRunResult) -> None:
        """将单样本结果写入日志文件，终端不输出。"""
        status = "CORRECT" if result.is_correct else ("WRONG" if result.is_completed else "INCOMPLETE")
        logger.info(
            "Sample %d: %s | rounds=%d | tokens=%dP/%dC | time=%.1fs",
            result.index, status, result.total_rounds,
            result.prompt_tokens, result.completion_tokens, result.elapsed_time,
        )

    def _update_and_save(self, summary: DatasetExperimentSummary) -> None:
        """更新汇总统计并增量保存（线程安全）。"""
        summary.accuracy = summary.correct_count / summary.total_samples if summary.total_samples > 0 else 0.0
        if summary.results:
            summary.avg_rounds = sum(r.total_rounds for r in summary.results) / len(summary.results)
            summary.avg_metric = sum(
                r.evaluation.metric_value for r in summary.results if r.evaluation
            ) / len(summary.results)
            if not summary.metric_name:
                for r in summary.results:
                    if r.evaluation and r.evaluation.metric_name:
                        summary.metric_name = r.evaluation.metric_name
                        break
        with self._save_lock:
            self._save_results(summary)

    def _run_single(self, sample: DatasetSample, index: int) -> DatasetRunResult:
        """运行单个样本。"""
        result = DatasetRunResult(
            index=index,
            question=sample.question,
            ground_truth=sample.ground_truth,
        )

        t_start = time.time()
        try:
            # 在 log_dir 下为每个样本创建独立的 trace 目录
            from mas_topo.experiment.topology_logger import TopologyLogger
            sample_trace_dir = self.log_dir / "trace" / f"sample_{index:03d}"
            topology_logger = TopologyLogger(
                trace_dir=str(sample_trace_dir),
                task=sample.question,
            )

            # 使用本地 tracker 记录本样本指标，避免并发覆盖计时状态
            local_tracker = ExperimentTracker()
            local_tracker.start_run()

            graph = Graph.from_config(self.config, topology_logger=topology_logger)
            graph_result = graph.run(
                sample.question,
                metadata={"test_cases": sample.ground_truth},
            )

            result.is_completed = graph_result.is_completed
            result.total_rounds = len(graph_result.rounds)
            result.prompt_tokens = graph_result.context.prompt_tokens
            result.completion_tokens = graph_result.context.completion_tokens
            result.final_answer = str(graph_result.final_answer) if graph_result.final_answer is not None else ""

            # 使用评估器进行统一评估（仅最终答案，不在执行过程中使用 ground-truth）
            eval_result = self.evaluator.evaluate(sample, result.final_answer)
            result.evaluation = eval_result
            result.is_correct = eval_result.is_correct
            result.predicted_answer = eval_result.predicted_answer

            # 记录指标到本地 tracker
            local_tracker.record_run(
                graph_result,
                task=sample.question[:200],
                sample_index=index,
                is_correct=result.is_correct,
                topology=getattr(self.config.topology, "strategy", ""),
                routing=getattr(self.config.routing, "strategy", ""),
                decision=getattr(self.config.decision, "strategy", ""),
            )

            # 合并到全局 tracker 并增量写入 JSONL
            metrics_jsonl_path = self.output_dir / "metrics.jsonl"
            with self._tracker_lock:
                self.tracker._runs.extend(local_tracker._runs)
                local_tracker.to_jsonl(str(metrics_jsonl_path), mode="a")

            # 完成单样本追踪
            topology_logger.finalize(graph_result)

        except Exception as e:
            logger.exception("Sample %d failed", index)
            result.error = str(e)

        result.elapsed_time = time.time() - t_start
        return result

    def _save_results(self, summary: DatasetExperimentSummary) -> dict:
        """导出实验结果到 JSON 和 CSV（不含过程追踪）。

        Returns:
            导出的文件路径字典。
        """
        # JSON 详细结果（仅结果，不含过程）
        json_path = self.output_dir / "results.json"
        json_path.write_text(
            json.dumps(summary.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # CSV 结果
        import csv
        csv_path = self.output_dir / "results.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index", "question", "ground_truth", "predicted_answer",
                "is_correct", "is_completed", "total_rounds",
                "prompt_tokens", "completion_tokens", "elapsed_time",
                "error",
            ])
            for r in summary.results:
                writer.writerow([
                    r.index, r.question, r.ground_truth, r.predicted_answer,
                    r.is_correct, r.is_completed, r.total_rounds,
                    r.prompt_tokens, r.completion_tokens, round(r.elapsed_time, 2),
                    r.error or "",
                ])

        # 汇总 JSON
        summary_path = self.output_dir / "summary.json"
        summary_dict = {
            "dataset_path": summary.dataset_path,
            "total_samples": summary.total_samples,
            "completed_count": summary.completed_count,
            "correct_count": summary.correct_count,
            "accuracy": round(summary.accuracy, 4),
            "metric_name": summary.metric_name,
            "avg_metric": round(summary.avg_metric, 4),
            "total_prompt_tokens": summary.total_prompt_tokens,
            "total_completion_tokens": summary.total_completion_tokens,
            "total_elapsed_time": round(summary.total_elapsed_time, 2),
            "avg_rounds": round(summary.avg_rounds, 2),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, ensure_ascii=False, indent=2)

        # 指标文件（JSON + CSV）
        metrics_json_path = self.output_dir / "metrics.json"
        metrics_csv_path = self.output_dir / "metrics.csv"
        self.tracker.to_json(str(metrics_json_path))
        self.tracker.to_csv(str(metrics_csv_path))

        paths = {
            "json": str(json_path),
            "csv": str(csv_path),
            "summary": str(summary_path),
            "metrics_json": str(metrics_json_path),
            "metrics_csv": str(metrics_csv_path),
        }

        logger.debug(
            "Results saved: json=%s, csv=%s, summary=%s, metrics_json=%s, metrics_csv=%s",
            paths["json"], paths["csv"], paths["summary"],
            paths["metrics_json"], paths["metrics_csv"],
        )

        return paths

    def print_summary(self, summary: DatasetExperimentSummary) -> None:
        """打印实验汇总。"""
        print("\n" + "=" * 60)
        print("Dataset Experiment Summary")
        print("=" * 60)
        print(f"  Dataset: {summary.dataset_path}")
        print(f"  Total samples: {summary.total_samples}")
        print(f"  Completed: {summary.completed_count}")
        print(f"  Correct: {summary.correct_count}")
        print(f"  Accuracy: {summary.accuracy:.2%}")
        if summary.metric_name:
            print(f"  Metric: {summary.metric_name} = {summary.avg_metric:.4f}")
        print(f"  Total tokens: {summary.total_prompt_tokens}P/{summary.total_completion_tokens}C")
        print(f"  Total time: {summary.total_elapsed_time:.1f}s")
        print(f"  Avg rounds: {summary.avg_rounds:.2f}")
        print("=" * 60)
