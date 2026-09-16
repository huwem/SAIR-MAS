"""MATH 数据集加载器。

加载 Hendrycks MATH 数据集（竞赛级数学推理），解析问题/解答/难度等级。
评估通过提取最终答案（\boxed{} 中的内容）进行精确匹配。
"""

import json
import re
from pathlib import Path
from typing import Optional

from mas_topo._registries import dataset_registry
from mas_topo.datasets.base import DatasetSample, BaseDatasetLoader


class MATHSample(DatasetSample):
    """单个 MATH 样本。"""

    def __init__(
        self,
        problem: str = "",
        solution: str = "",
        level: str = "",
        problem_type: str = "",
        index: int = 0,
    ):
        self.problem = problem
        self.solution = solution
        self.level = level
        self.problem_type = problem_type
        self.index = index

    @property
    def question(self) -> str:
        return self.problem

    @property
    def ground_truth(self) -> str:
        """从 solution 中提取 \boxed{} 中的最终答案。"""
        return self._extract_boxed_answer(self.solution)

    @staticmethod
    def _extract_boxed_answer(solution: str) -> str:
        """从 LaTeX solution 中提取 \boxed{...} 中的答案。"""
        if not solution:
            return ""
        # 匹配 \boxed{...}，支持嵌套花括号（如 \frac{\pi}{2}、\sqrt{3} 等）
        pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
        matches = re.findall(pattern, solution)
        if matches:
            return matches[-1].strip()  # 取最后一个（通常是最终答案）
        # 回退：提取 #### 后的数字（GSM8K 风格）
        hash_match = re.search(r"####\s*(\S+)", solution)
        if hash_match:
            return hash_match.group(1).strip()
        return solution.strip()[-50:]  # 回退：取末尾50字符

    def check_answer(self, predicted: str) -> bool:
        """检查预测答案是否正确。"""
        if not predicted:
            return False
        predicted = self._normalize_answer(predicted)
        ground = self._normalize_answer(self.ground_truth)
        return predicted == ground

    @staticmethod
    def _normalize_answer(text: str) -> str:
        """标准化答案文本：去除空格、统一分数格式等。"""
        if not text:
            return ""
        text = str(text).strip().lower()
        # 去除美元符号和单位
        text = re.sub(r"\\?\$", "", text)
        text = re.sub(r"\\text\{[^}]*\}", "", text)
        # 去除 LaTeX 空格命令
        text = re.sub(r"\\[;,]", "", text)
        # 统一分数
        text = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", text)
        # 去除多余空格
        text = re.sub(r"\s+", "", text)
        return text

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "problem": self.problem,
            "solution": self.solution,
            "level": self.level,
            "type": self.problem_type,
            "ground_truth": self.ground_truth,
        }


@dataset_registry.register(
    "math",
    display_name="MATH 竞赛数学数据集",
    description="Hendrycks MATH 数据集，包含 AMC/AIME 等竞赛级数学问题，"
    "分 5 个难度等级（Level 1-5）。",
)
class MATHLoader(BaseDatasetLoader):
    """MATH 数据集加载器。

    期望数据格式：
    - data_path 指向一个目录，包含 train/test 子目录
    - 每个子目录下有按题型分类的文件夹（如 algebra, geometry 等）
    - 每个题目是一个 JSON 文件，包含 problem/solution/level/type

    或单个 JSONL 文件，每行一个题目。
    """

    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.data_path = Path(data_path)

    def load(self, limit: Optional[int] = None) -> list[MATHSample]:
        self._samples = []
        
        if self.data_path.is_file() and self.data_path.suffix == ".jsonl":
            # JSONL 格式
            self._load_jsonl(limit)
        elif self.data_path.is_dir():
            # 原始目录格式
            self._load_directory(limit)
        else:
            raise FileNotFoundError(
                f"MATH dataset not found: {self.data_path}\n"
                "Expected a .jsonl file or a directory with test/ subdir."
            )
        
        return self._samples

    def _load_jsonl(self, limit: Optional[int] = None):
        """从 JSONL 文件加载。"""
        with self.data_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                obj = json.loads(line.strip())
                sample = MATHSample(
                    problem=obj.get("problem", ""),
                    solution=obj.get("solution", ""),
                    level=str(obj.get("level", "")),
                    problem_type=obj.get("type", ""),
                    index=idx,
                )
                self._samples.append(sample)

    def _load_directory(self, limit: Optional[int] = None):
        """从原始目录结构加载。"""
        test_dir = self.data_path / "test"
        if not test_dir.exists():
            test_dir = self.data_path / "train"
        
        if not test_dir.exists():
            raise FileNotFoundError(f"No test/ or train/ directory found in {self.data_path}")
        
        idx = 0
        for category_dir in sorted(test_dir.iterdir()):
            if not category_dir.is_dir():
                continue
            for problem_file in sorted(category_dir.glob("*.json")):
                if limit is not None and idx >= limit:
                    return
                with problem_file.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                sample = MATHSample(
                    problem=obj.get("problem", ""),
                    solution=obj.get("solution", ""),
                    level=str(obj.get("level", "")),
                    problem_type=obj.get("type", category_dir.name),
                    index=idx,
                )
                self._samples.append(sample)
                idx += 1

    def total_count(self) -> int:
        if not self.data_path.exists():
            return 0
        if self.data_path.is_file():
            count = 0
            with self.data_path.open("r", encoding="utf-8") as f:
                for _ in f:
                    count += 1
            return count
        else:
            test_dir = self.data_path / "test"
            if not test_dir.exists():
                return 0
            count = 0
            for category_dir in test_dir.iterdir():
                if category_dir.is_dir():
                    count += len(list(category_dir.glob("*.json")))
            return count
