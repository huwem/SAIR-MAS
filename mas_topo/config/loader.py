"""配置文件加载器。

支持 YAML/JSON 格式，支持 ${ENV_VAR} 环境变量替换。
基于 Pydantic 模型自动验证配置字段。

支持两种配置模式（自动检测）：
- 实验级配置（ExperimentConfig）：包含 experiment 顶层字段，定义数据集+评估+框架
- 框架级配置（FrameworkConfig）：仅定义多智能体框架参数（向后兼容）
"""

import json
import os
from pathlib import Path
from typing import Union

from mas_topo.config.schema import FrameworkConfig
from mas_topo.config.experiment_schema import ExperimentConfig


def load_config(path: str) -> Union[ExperimentConfig, FrameworkConfig]:
    """从 YAML 或 JSON 文件加载配置。

    自动检测配置类型：
    - 若文件包含顶层 `experiment` 或 `framework` 字段，解析为 ExperimentConfig
    - 否则解析为 FrameworkConfig（向后兼容）

    支持 ${ENV_VAR} 形式的环境变量替换。
    """
    raw = _read_file(path)
    raw = _substitute_env_vars(raw)
    return _parse_config(raw)


def load_config_from_dict(data: dict) -> Union[ExperimentConfig, FrameworkConfig]:
    """从字典加载配置。

    自动检测配置类型，同 load_config。
    """
    data = _substitute_env_vars(data)
    return _parse_config(data)


def _parse_config(data: dict) -> Union[ExperimentConfig, FrameworkConfig]:
    """根据字典结构自动选择配置模型。"""
    if not isinstance(data, dict):
        raise TypeError(f"Config must be a dict, got {type(data)}")

    # 检测实验级配置：有 experiment 顶层字段，或同时有 dataset + framework 字段
    if "experiment" in data:
        # 嵌套结构: {experiment: {dataset: ..., framework: ...}}
        return ExperimentConfig.from_dict(data["experiment"])

    if "framework" in data and ("dataset" in data or "evaluation" in data):
        # 扁平结构: {dataset: ..., framework: ..., evaluation: ...}
        return ExperimentConfig.from_dict(data)

    # 否则当作纯框架配置（向后兼容）
    return FrameworkConfig.from_dict(data)


def _read_file(path: str) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(text) or {}
        except ImportError:
            raise ImportError("PyYAML required for YAML config: pip install pyyaml")
    elif p.suffix == ".json":
        return json.loads(text)
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")


def _substitute_env_vars(obj):
    """递归替换 ${ENV_VAR} 为环境变量值。"""
    if isinstance(obj, str):
        if obj.startswith("${") and obj.endswith("}"):
            var = obj[2:-1]
            return os.environ.get(var, "")
        return obj
    if isinstance(obj, dict):
        return {k: _substitute_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env_vars(v) for v in obj]
    return obj
