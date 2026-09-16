"""实验配置数据模型。

定义 ExperimentConfig，封装数据集、评估指标、输出格式等实验级参数，
与 FrameworkConfig（框架级参数）分离，实现真正的配置驱动实验。
"""

from typing import Optional

from pydantic import BaseModel, Field

from mas_topo.config.schema import FrameworkConfig


class DatasetConfig(BaseModel):
    """数据集配置。"""

    loader: str = Field(default="humaneval", description="数据集加载器注册表名")
    path: str = Field(default="", description="数据集文件路径")
    limit: Optional[int] = Field(default=None, description="最多运行的样本数")
    offset: int = Field(default=0, ge=0, description="起始偏移量")
    random_offset: bool = Field(default=False, description="是否随机选择起始偏移")
    seed: Optional[int] = Field(default=None, description="随机种子")
    sample_timeout: float = Field(
        default=0.0, ge=0.0,
        description="单样本墙钟超时（秒），0 表示不限制；超时样本记为 error 并继续",
    )


class EvaluationConfig(BaseModel):
    """评估配置。"""

    metric: str = Field(default="exact_match", description="评估器注册表名")
    metric_params: dict = Field(default_factory=dict, description="评估器额外参数")


class OutputConfig(BaseModel):
    """输出配置。"""

    dir: Optional[str] = Field(default=None, description="输出目录（None 则自动生成）")
    formats: list[str] = Field(default_factory=lambda: ["json", "csv"], description="输出格式列表")
    trace: bool = Field(default=True, description="是否启用过程追踪")


class ExperimentConfig(BaseModel):
    """实验总配置。

    通过单个 YAML/JSON 文件定义完整实验：数据集 + 评估 + 框架。
    """

    name: Optional[str] = Field(default=None, description="实验名称")
    description: Optional[str] = Field(default=None, description="实验描述")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    framework: FrameworkConfig = Field(default_factory=FrameworkConfig)

    def to_dict(self) -> dict:
        """导出为字典。"""
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        """从字典构建。"""
        return cls.model_validate(data)
