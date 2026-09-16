"""批量实验运行器。

支持交叉组合多种拓扑/路由/决策策略，批量运行实验并跟踪指标。
自动在 config.log_dir 下创建过程追踪（trace）和结果（results）文件。
"""

import itertools
import os
from pathlib import Path
from typing import Optional

from mas_topo.config.schema import FrameworkConfig
from mas_topo.graph import Graph
from mas_topo.experiment.tracker import ExperimentTracker, RunMetrics
from mas_topo.experiment.topology_logger import write_results_file


class ExperimentRunner:
    """批量实验运行器。

    支持多配置 × 多任务交叉组合，自动记录指标，
    并将过程追踪与结果写入独立文件。

    文件输出（当 config.log_dir 设置时）:
        {log_dir}/
        ├── trace/            # 过程追踪（每轮 Agent I/O、拓扑、路由决策、计时）
        │   ├── trace.json
        │   ├── round_*.md
        │   └── index.md
        ├── results.json      # 实验结果（最终答案、指标汇总）
        └── results.csv       # CSV 格式结果
    """

    def __init__(self, verbose: bool = True, output_dir: Optional[str] = None):
        self.tracker = ExperimentTracker()
        self.verbose = verbose
        self.output_dir = output_dir

    def run(
        self,
        configs: list[FrameworkConfig],
        tasks: list[str],
    ) -> list[RunMetrics]:
        """批量运行实验：每个 config × 每个 task 组合运行一次。

        Args:
            configs: FrameworkConfig 列表。
            tasks: 任务描述列表。

        Returns:
            所有运行的指标列表。
        """
        results: list[RunMetrics] = []

        for cfg, task in itertools.product(configs, tasks):
            if self.verbose:
                topo = cfg.topology.strategy
                rout = cfg.routing.strategy
                dec = cfg.decision.strategy
                agents = [a.name for a in cfg.agents]
                print(f"\n[Experiment] topology={topo} routing={rout} "
                      f"decision={dec} agents={agents}")
                print(f"  Task: {task[:80]}...")

            # from_config 会在 config.log_dir 已设置时自动创建 TopologyLogger
            graph = Graph.from_config(cfg)
            self.tracker.start_run()
            graph_result = graph.run(task)

            metrics = self.tracker.record_run(
                graph_result,
                task=task,
                topology=cfg.topology.strategy,
                routing=cfg.routing.strategy,
                decision=cfg.decision.strategy,
            )
            results.append(metrics)

            if self.verbose:
                status = "COMPLETED" if graph_result.is_completed else "TIMEOUT"
                print(f"  Result: {status} in {metrics.total_rounds} rounds "
                      f"({metrics.elapsed_time:.1f}s)")

        # 自动写入结果文件
        self._save_results(results)
        return results

    async def arun(
        self,
        configs: list[FrameworkConfig],
        tasks: list[str],
    ) -> list[RunMetrics]:
        """异步批量运行实验。"""
        results: list[RunMetrics] = []

        for cfg, task in itertools.product(configs, tasks):
            if self.verbose:
                print(f"[Experiment] topology={cfg.topology.strategy} "
                      f"routing={cfg.routing.strategy} task={task[:60]}...")

            graph = Graph.from_config(cfg)
            self.tracker.start_run()
            graph_result = await graph.arun(task)

            metrics = self.tracker.record_run(
                graph_result,
                task=task,
                topology=cfg.topology.strategy,
                routing=cfg.routing.strategy,
                decision=cfg.decision.strategy,
            )
            results.append(metrics)

        self._save_results(results)
        return results

    def _save_results(self, results: list[RunMetrics]) -> None:
        """将实验结果写入独立的结果文件。"""
        # 确定输出目录
        if self.output_dir:
            out_dir = Path(self.output_dir)
        elif results:
            # 尝试从第一个 config 获取 log_dir
            out_dir = Path("experiment_results")
        else:
            return

        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON 结果
        json_path = out_dir / "results.json"
        self.tracker.to_json(str(json_path))

        # CSV 结果
        csv_path = out_dir / "results.csv"
        self.tracker.to_csv(str(csv_path))

        if self.verbose:
            print(f"\n[Results saved]")
            print(f"  JSON: {json_path}")
            print(f"  CSV:  {csv_path}")

    def get_tracker(self) -> ExperimentTracker:
        """获取指标跟踪器。"""
        return self.tracker

    def compare_topologies(
        self,
        topology_names: list[str],
        base_config: FrameworkConfig,
        tasks: list[str],
    ) -> list[RunMetrics]:
        """比较不同拓扑策略在相同配置下的表现。

        Args:
            topology_names: 要比较的拓扑策略名称列表。
            base_config: 基础配置（topology 字段会被覆盖）。
            tasks: 任务列表。

        Returns:
            所有运行的指标列表。
        """
        configs = []
        for topo_name in topology_names:
            cfg = FrameworkConfig(
                agents=base_config.agents,
                topology=base_config.topology.__class__(
                    **{**base_config.topology.__dict__, "strategy": topo_name}
                ),
                decision=base_config.decision,
                routing=base_config.routing,
                llm=base_config.llm,
                max_rounds=base_config.max_rounds,
                verbose=base_config.verbose,
            )
            configs.append(cfg)
        return self.run(configs, tasks)

    def compare_routings(
        self,
        routing_names: list[str],
        base_config: FrameworkConfig,
        tasks: list[str],
    ) -> list[RunMetrics]:
        """比较不同路由策略在相同配置下的表现。"""
        configs = []
        for rout_name in routing_names:
            cfg = FrameworkConfig(
                agents=base_config.agents,
                topology=base_config.topology,
                decision=base_config.decision,
                routing=base_config.routing.__class__(
                    **{**base_config.routing.__dict__, "strategy": rout_name}
                ),
                llm=base_config.llm,
                max_rounds=base_config.max_rounds,
                verbose=base_config.verbose,
            )
            configs.append(cfg)
        return self.run(configs, tasks)
