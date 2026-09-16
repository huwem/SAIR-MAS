"""HumanEval 数据集加载器。

加载 HumanEval 数据集，解析 prompt/test，通过执行测试评估代码正确性。
代码提取和执行委托给 mas_topo.tools.builtin 中的可复用模块。
"""

import json
from pathlib import Path
from typing import Optional

from mas_topo._registries import dataset_registry
from mas_topo.datasets.base import DatasetSample, BaseDatasetLoader
from mas_topo.tools.builtin import extract_python_code, run_python_tests


class HumanEvalSample(DatasetSample):
    """单个 HumanEval 样本。"""

    def __init__(
        self,
        task_id: str = "",
        prompt: str = "",
        test: str = "",
        entry_point: str = "",
        index: int = 0,
    ):
        self.task_id = task_id
        self.prompt = prompt
        self.test = test
        self.entry_point = entry_point
        self.index = index

    @property
    def question(self) -> str:
        return (
            f"Implement the following Python function:\n\n"
            f"```python\n{self.prompt}\n```"
        )

    @property
    def ground_truth(self) -> str:
        return self.test

    def check_answer(self, predicted: str) -> bool:
        """通过执行测试检查预测代码是否通过（pass@1 评估）。"""
        if not predicted or not predicted.strip():
            return False

        code = extract_python_code(predicted)
        if not code:
            if "def " not in predicted and "class " not in predicted:
                return False
            code = predicted.strip()

        result = run_python_tests(code=code, tests=self.test)
        return result.get("passed", False)

    @staticmethod
    def _extract_number(text: str) -> str:
        """兼容接口：从文本中提取代码摘要用于结果展示。"""
        if not text:
            return ""
        code = extract_python_code(text)
        if code:
            lines = code.strip().split("\n")
            return "\n".join(lines[:5]) + ("\n..." if len(lines) > 5 else "")
        return text[:200]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "test": self.test,
            "entry_point": self.entry_point,
        }


@dataset_registry.register(
    "humaneval",
    display_name="HumanEval 代码生成数据集",
    description="OpenAI HumanEval 数据集，用于评估多智能体代码生成能力（pass@1）。",
)
class HumanEvalLoader(BaseDatasetLoader):
    """HumanEval 数据集加载器。"""

    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.data_path = Path(data_path)

    def load(self, limit: Optional[int] = None) -> list[HumanEvalSample]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self._samples = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                obj = json.loads(line.strip())
                sample = HumanEvalSample(
                    task_id=obj.get("name", f"HumanEval_{idx}"),
                    prompt=obj.get("prompt", ""),
                    test=obj.get("test", ""),
                    entry_point=obj.get("entry_point", ""),
                    index=idx,
                )
                self._samples.append(sample)

        return self._samples

    def total_count(self) -> int:
        if not self.data_path.exists():
            return 0
        count = 0
        with self.data_path.open("r", encoding="utf-8") as f:
            for _ in f:
                count += 1
        return count
