"""评估器抽象基类。

定义统一的 Evaluator 接口，支持通过注册表按名称动态选择和组合评估指标。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from mas_topo.datasets.base import DatasetSample


@dataclass
class EvaluationResult:
    """单个样本的评估结果。"""

    is_correct: bool = False
    predicted_answer: str = ""
    metric_value: float = 0.0           # 连续指标值（如 BLEU、ROUGE）
    metric_name: str = ""               # 指标名称
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_correct": self.is_correct,
            "predicted_answer": self.predicted_answer,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "details": self.details,
        }


class Evaluator(ABC):
    """评估器抽象基类。

    负责将 Agent 的预测输出与样本 ground_truth 对比，生成 EvaluationResult。
    支持通过注册表动态发现和加载。
    """

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def evaluate(self, sample: DatasetSample, predicted: str) -> EvaluationResult:
        """评估单个样本的预测结果。

        Args:
            sample: 数据集样本（含 ground_truth）。
            predicted: Agent 生成的预测文本（如 final_answer）。

        Returns:
            EvaluationResult 评估结果。
        """

    def summarize(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """汇总一批评估结果。

        默认实现计算准确率和平均 metric_value。
        子类可覆盖以提供更复杂的汇总逻辑（如 pass@k、BLEU 平均等）。
        """
        total = len(results)
        if total == 0:
            return {"accuracy": 0.0, "avg_metric": 0.0, "total": 0}

        correct = sum(1 for r in results if r.is_correct)
        avg_metric = sum(r.metric_value for r in results) / total

        return {
            "accuracy": correct / total,
            "correct_count": correct,
            "avg_metric": avg_metric,
            "total": total,
        }
