"""AIME 数据集加载器。

加载 AIME（美国数学邀请赛）数据集，答案为 0-999 的整数。
评估通过提取数字进行精确匹配。
"""

import json
import re
from pathlib import Path
from typing import Optional

from mas_topo._registries import dataset_registry
from mas_topo.datasets.base import DatasetSample, BaseDatasetLoader


class AIMESample(DatasetSample):
    """单个 AIME 样本。"""

    def __init__(
        self,
        problem: str = "",
        answer: int = 0,
        index: int = 0,
    ):
        self.problem = problem
        self.answer = answer
        self.index = index

    @property
    def question(self) -> str:
        return self.problem

    @property
    def ground_truth(self) -> str:
        return str(self.answer)

    def check_answer(self, predicted: str) -> bool:
        """从预测文本中提取整数答案并与 ground truth 比较。"""
        if not predicted:
            return False
        # 提取所有整数
        numbers = re.findall(r"-?\d+", str(predicted))
        if not numbers:
            return False
        # 取最后一个整数（通常是最终答案）
        return int(numbers[-1]) == self.answer

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "problem": self.problem[:200],
            "answer": self.answer,
        }


@dataset_registry.register(
    "aime",
    display_name="AIME 数学竞赛数据集",
    description="AIME（美国数学邀请赛）2024-2026，答案为 0-999 整数，"
    "适用 exact_match 评估。",
)
class AIMELoader(BaseDatasetLoader):
    """AIME 数据集加载器。

    期望 JSONL 格式，每行一个题目，包含 problem（LaTeX）/ answer（整数）。
    支持通过 glob 模式加载多年份数据（如 datasets/aime/aime_*.jsonl）。
    """

    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.data_path = Path(data_path)

    def _resolve_paths(self) -> list[Path]:
        """解析数据路径，支持 glob 模式。"""
        if "*" in str(self.data_path) or "?" in str(self.data_path):
            parent = self.data_path.parent
            pattern = self.data_path.name
            if parent.exists():
                return sorted(parent.glob(pattern))
            return []
        if self.data_path.exists():
            return [self.data_path]
        return []

    def load(self, limit: Optional[int] = None) -> list[AIMESample]:
        paths = self._resolve_paths()
        if not paths:
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self._samples = []
        idx = 0
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    if limit is not None and idx >= limit:
                        return self._samples
                    obj = json.loads(line.strip())
                    sample = AIMESample(
                        problem=obj.get("problem", ""),
                        answer=int(obj.get("answer", 0)),
                        index=idx,
                    )
                    self._samples.append(sample)
                    idx += 1

        return self._samples

    def total_count(self) -> int:
        paths = self._resolve_paths()
        count = 0
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                count += sum(1 for _ in f)
        return count
