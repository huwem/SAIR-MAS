"""LiveCodeBench 数据集加载器。

加载 LiveCodeBench 的 JSONL 数据，支持公开/私有 stdin/stdout 测试用例。
"""

import base64
import json
import zlib
from pathlib import Path
from typing import Any, Optional

from mas_topo._registries import dataset_registry
from mas_topo.datasets.base import DatasetSample, BaseDatasetLoader


class LiveCodeBenchSample(DatasetSample):
    """单个 LiveCodeBench 样本。"""

    def __init__(
        self,
        question_id: str = "",
        title: str = "",
        content: str = "",
        public_tests: Optional[list[dict]] = None,
        private_tests: Optional[list[dict]] = None,
        platform: str = "",
        index: int = 0,
    ):
        self.question_id = question_id
        self.title = title
        self.content = content
        self.public_tests = public_tests or []
        self.private_tests = private_tests or []
        self.platform = platform
        self.index = index

    @property
    def question(self) -> str:
        return self.content

    @property
    def ground_truth(self) -> str:
        """只返回可公开的测试；运行器会将此字段传入 agent 上下文。"""
        return json.dumps(self.public_tests)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "question_id": self.question_id,
            "title": self.title,
            "public_tests": self.public_tests,
            "private_tests": self.private_tests,
            "platform": self.platform,
        }

    def _extract_number(self, text: str) -> str:
        """兼容接口：取前 200 字符摘要。"""
        return str(text)[:200] if text else ""


def _decode_test_cases(raw: Any) -> list[dict]:
    """解码测试用例字段。

    支持三种形式：
    1. 已是 list[dict]
    2. JSON 字符串
    3. base64(zlib(pickle(json_str))) 编码字符串（LiveCodeBench 私有用例）
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        data = raw
    elif isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        # 尝试 JSON 字符串
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                compressed = base64.b64decode(raw, validate=True)
                decompressed = zlib.decompress(compressed)
                try:
                    data = json.loads(decompressed)
                except (ValueError, UnicodeDecodeError):
                    import pickle
                    data = json.loads(pickle.loads(decompressed))
            except Exception as exc:
                raise ValueError("Cannot decode test cases") from exc
    else:
        raise ValueError("Test cases must be a list or encoded string")
    if not isinstance(data, list) or any(
        not isinstance(case, dict) or "input" not in case or "output" not in case
        for case in data
    ):
        raise ValueError("Test cases must contain input/output objects")
    return data


@dataset_registry.register(
    "livecodebench",
    display_name="LiveCodeBench 代码生成数据集",
    description="LiveCodeBench 竞赛编程数据集，基于 stdin/stdout 测试用例评估代码正确性。",
)
class LiveCodeBenchLoader(BaseDatasetLoader):
    """LiveCodeBench 数据集加载器。"""

    def __init__(self, data_path: str):
        super().__init__(data_path)
        self.data_path = Path(data_path)

    def load(self, limit: Optional[int] = None) -> list[LiveCodeBenchSample]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        self._samples = []
        with self.data_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if limit is not None and idx >= limit:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line.strip())
                decoded = {}
                for field in ("public_test_cases", "private_test_cases"):
                    try:
                        raw = obj.get(field)
                        if field == "private_test_cases" and (
                            raw is None or isinstance(raw, str) and not raw.strip()
                        ):
                            raise ValueError("Missing private test suite")
                        decoded[field] = _decode_test_cases(raw)
                    except ValueError as exc:
                        raise ValueError(
                            f"{self.data_path}: line {idx + 1}: invalid {field}: {exc}"
                        ) from exc
                sample = LiveCodeBenchSample(
                    question_id=obj.get("question_id", f"lcb_{idx}"),
                    title=obj.get("question_title", ""),
                    content=obj.get("question_content", ""),
                    public_tests=decoded["public_test_cases"],
                    private_tests=decoded["private_test_cases"],
                    platform=obj.get("platform", ""),
                    index=idx,
                )
                self._samples.append(sample)

        return self._samples

    def total_count(self) -> int:
        if not self.data_path.exists():
            return 0
        count = 0
        with self.data_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
