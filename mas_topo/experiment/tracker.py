"""实验指标跟踪与结果导出。

跟踪拓扑指标、通信指标、性能指标，支持 JSON/JSONL/CSV 导出，
便于后续数据分析与可视化。
"""

import csv
import io
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class RoundMetrics:
    """单轮指标。"""

    round_num: int = 0
    num_edges: int = 0
    graph_density: float = 0.0
    avg_in_degree: float = 0.0
    avg_out_degree: float = 0.0
    num_messages: int = 0
    avg_message_length: float = 0.0
    elapsed_time: float = 0.0
    is_complete: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    active_agents: list[str] = field(default_factory=list)
    illocution_counts: dict[str, int] = field(default_factory=dict)
    channel_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class RunMetrics:
    """单次运行指标。"""

    task: str = ""
    sample_index: int = -1
    total_rounds: int = 0
    is_completed: bool = False
    is_correct: bool = False
    final_answer: str = ""
    total_cost: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    elapsed_time: float = 0.0
    topology_strategy: str = ""
    routing_strategy: str = ""
    decision_strategy: str = ""
    round_metrics: list[RoundMetrics] = field(default_factory=list)


class ExperimentTracker:
    """实验指标跟踪器。

    从 GraphResult 中提取各级指标并支持导出。
    """

    def __init__(self):
        self._runs: list[RunMetrics] = []
        self._start_time: Optional[float] = None

    def start_run(self) -> None:
        """开始计时。"""
        self._start_time = time.time()

    def record_run(
        self,
        graph_result,
        task: str = "",
        sample_index: int = -1,
        is_correct: bool = False,
        topology: str = "",
        routing: str = "",
        decision: str = "",
    ) -> RunMetrics:
        """从 GraphResult 记录一次运行的指标。"""
        elapsed = time.time() - self._start_time if self._start_time else 0.0
        self._start_time = None

        ctx = graph_result.context
        run = RunMetrics(
            task=task,
            sample_index=sample_index,
            total_rounds=len(graph_result.rounds),
            is_completed=graph_result.is_completed,
            is_correct=is_correct,
            final_answer=graph_result.final_answer or "",
            total_cost=ctx.cost,
            total_prompt_tokens=ctx.prompt_tokens,
            total_completion_tokens=ctx.completion_tokens,
            elapsed_time=elapsed,
            topology_strategy=topology,
            routing_strategy=routing,
            decision_strategy=decision,
        )

        for r in graph_result.rounds:
            adj = r.adjacency_matrix
            if adj is not None:
                N = adj.shape[0]
                num_edges = int(adj.sum())
                density = num_edges / (N * (N - 1)) if N > 1 else 0.0
                avg_in = float(adj.sum(axis=0).mean()) if N > 0 else 0.0
                avg_out = float(adj.sum(axis=1).mean()) if N > 0 else 0.0
            else:
                N = 0
                num_edges = 0
                density = 0.0
                avg_in = 0.0
                avg_out = 0.0

            msgs = r.routed_messages
            num_msgs = sum(len(v) for v in msgs.values())
            lengths = []
            channel_counts: Counter = Counter()
            for msg_list in msgs.values():
                for msg in msg_list:
                    lengths.append(len(getattr(msg, "content", "")))
                    channel_counts[getattr(msg, "channel", "default")] += 1
            avg_len = float(np.mean(lengths)) if lengths else 0.0

            # 统计本轮激活的 Agent 与语旨分布
            active_agents: list[str] = []
            illocution_counts: Counter = Counter()
            for name, out in (r.agent_outputs or {}).items():
                if out is None:
                    continue
                active_agents.append(name)
                for item in (getattr(out, "private_routing", None) or []):
                    if isinstance(item, dict):
                        illoc = item.get("illocution")
                        if illoc:
                            illocution_counts[str(illoc)] += 1

            run.round_metrics.append(RoundMetrics(
                round_num=r.round_num,
                num_edges=num_edges,
                graph_density=round(density, 4),
                avg_in_degree=round(avg_in, 2),
                avg_out_degree=round(avg_out, 2),
                num_messages=num_msgs,
                avg_message_length=round(avg_len, 1),
                elapsed_time=round(r.elapsed_time, 3),
                is_complete=r.decision.is_complete if r.decision else False,
                prompt_tokens=getattr(ctx, "prompt_tokens", 0),
                completion_tokens=getattr(ctx, "completion_tokens", 0),
                active_agents=active_agents,
                illocution_counts=dict(illocution_counts),
                channel_counts=dict(channel_counts),
            ))

        self._runs.append(run)
        return run

    def get_runs(self) -> list[RunMetrics]:
        """获取所有记录的运行指标。"""
        return self._runs

    def to_json(self, path: Optional[str] = None) -> str:
        """导出为 JSON 字符串或写入文件。"""
        data = [self._run_to_dict(run) for run in self._runs]
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            Path(path).write_text(json_str, encoding="utf-8")
        return json_str

    def to_jsonl(self, path: Optional[str] = None, mode: str = "a") -> str:
        """导出为 JSON Lines 格式或追加到文件。

        每行对应一次运行（一个样本），便于增量采集与流式读取。
        """
        lines = [json.dumps(self._run_to_dict(run), ensure_ascii=False) for run in self._runs]
        text = "\n".join(lines)
        if text:
            text += "\n"
        if path:
            with open(path, mode, encoding="utf-8") as f:
                f.write(text)
        return text

    def to_csv(self, path: Optional[str] = None) -> str:
        """导出为 CSV 字符串或写入文件（每行对应一轮）。"""
        output = io.StringIO()
        writer = csv.writer(output)

        writer.writerow([
            "sample_index", "task", "total_rounds", "is_completed", "is_correct",
            "total_cost", "total_prompt_tokens", "total_completion_tokens", "elapsed_time",
            "topology", "routing", "decision",
            "round_num", "num_edges", "graph_density",
            "avg_in_degree", "avg_out_degree",
            "num_messages", "avg_message_length", "round_elapsed_time",
            "active_agents", "illocution_counts", "channel_counts",
        ])

        for run in self._runs:
            for rm in run.round_metrics:
                writer.writerow([
                    run.sample_index, run.task, run.total_rounds, run.is_completed, run.is_correct,
                    run.total_cost, run.total_prompt_tokens, run.total_completion_tokens,
                    run.elapsed_time, run.topology_strategy, run.routing_strategy,
                    run.decision_strategy, rm.round_num, rm.num_edges, rm.graph_density,
                    rm.avg_in_degree, rm.avg_out_degree, rm.num_messages,
                    rm.avg_message_length, rm.elapsed_time,
                    ";".join(rm.active_agents),
                    json.dumps(rm.illocution_counts, ensure_ascii=False),
                    json.dumps(rm.channel_counts, ensure_ascii=False),
                ])
            if not run.round_metrics:
                writer.writerow([
                    run.sample_index, run.task, run.total_rounds, run.is_completed, run.is_correct,
                    run.total_cost, run.total_prompt_tokens, run.total_completion_tokens,
                    run.elapsed_time, run.topology_strategy, run.routing_strategy,
                    run.decision_strategy,
                    "", "", "", "", "", "", "", "", "", "",
                ])

        csv_str = output.getvalue()
        if path:
            Path(path).write_text(csv_str, encoding="utf-8")
        return csv_str

    def summary(self) -> str:
        """生成人类可读的汇总报告。"""
        lines = ["=" * 60, "Experiment Summary", "=" * 60]
        for i, run in enumerate(self._runs):
            lines.append(f"\nRun {i+1}: {run.task}")
            lines.append(f"  Topology: {run.topology_strategy}")
            lines.append(f"  Routing:  {run.routing_strategy}")
            lines.append(f"  Decision: {run.decision_strategy}")
            lines.append(f"  Completed: {run.is_completed} in {run.total_rounds} rounds")
            lines.append(f"  Correct: {run.is_correct}")
            lines.append(f"  Time: {run.elapsed_time:.1f}s | "
                         f"Tokens: {run.total_prompt_tokens}P/"
                         f"{run.total_completion_tokens}C")
            lines.append(f"  Answer: {run.final_answer[:100]}...")
        return "\n".join(lines)

    @staticmethod
    def _run_to_dict(run: RunMetrics) -> dict:
        """将 RunMetrics 转为字典。"""
        return {
            "sample_index": run.sample_index,
            "task": run.task,
            "total_rounds": run.total_rounds,
            "is_completed": run.is_completed,
            "is_correct": run.is_correct,
            "final_answer": run.final_answer,
            "total_cost": run.total_cost,
            "total_prompt_tokens": run.total_prompt_tokens,
            "total_completion_tokens": run.total_completion_tokens,
            "elapsed_time": run.elapsed_time,
            "topology_strategy": run.topology_strategy,
            "routing_strategy": run.routing_strategy,
            "decision_strategy": run.decision_strategy,
            "rounds": [
                {
                    "round_num": rm.round_num,
                    "num_edges": rm.num_edges,
                    "graph_density": rm.graph_density,
                    "avg_in_degree": rm.avg_in_degree,
                    "avg_out_degree": rm.avg_out_degree,
                    "num_messages": rm.num_messages,
                    "avg_message_length": rm.avg_message_length,
                    "elapsed_time": rm.elapsed_time,
                    "is_complete": rm.is_complete,
                    "prompt_tokens": rm.prompt_tokens,
                    "completion_tokens": rm.completion_tokens,
                    "active_agents": rm.active_agents,
                    "illocution_counts": rm.illocution_counts,
                    "channel_counts": rm.channel_counts,
                }
                for rm in run.round_metrics
            ],
        }
