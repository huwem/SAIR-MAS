"""数据集加载器抽象基类。

定义统一的 DatasetSample 和 BaseDatasetLoader 接口，
支持通过注册表动态发现和加载数据集。
"""

from abc import ABC, abstractmethod
from typing import Optional


class DatasetSample(ABC):
    """数据集样本抽象基类。

    所有具体样本类必须实现 question、ground_truth 属性和 to_dict 方法。
    check_answer 和 _extract_number 为可选的评估辅助方法。
    """

    @property
    @abstractmethod
    def question(self) -> str:
        """返回任务/问题文本。"""

    @property
    @abstractmethod
    def ground_truth(self) -> str:
        """返回标准答案文本。"""

    @abstractmethod
    def to_dict(self) -> dict:
        """序列化为字典。"""

    def check_answer(self, predicted: str) -> bool:
        """检查预测答案是否正确（默认实现为字符串比较）。

        子类可覆盖以提供领域特定的评估逻辑。
        """
        if predicted is None:
            return False
        return str(predicted).strip() == str(self.ground_truth).strip()

    def _extract_number(self, text: str) -> str:
        """从文本中提取数字或摘要（兼容接口）。

        供 DatasetExperimentRunner 在结果展示时使用。
        子类可覆盖以提供领域特定的提取逻辑。
        """
        return str(text)[:200] if text else ""


class BaseDatasetLoader(ABC):
    """数据集加载器抽象基类。

    统一接口支持通过注册表按名称动态加载任意数据集。
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self._samples: list[DatasetSample] = []

    @abstractmethod
    def load(self, limit: Optional[int] = None) -> list[DatasetSample]:
        """加载数据集样本。

        Args:
            limit: 最多加载的样本数，None 表示全部加载。

        Returns:
            DatasetSample 列表。
        """

    def get_samples(self) -> list[DatasetSample]:
        """获取已缓存的样本。"""
        return self._samples

    @abstractmethod
    def total_count(self) -> int:
        """统计数据文件中的样本总数（不解析内容，仅计数）。"""

    def __len__(self) -> int:
        return len(self._samples)
