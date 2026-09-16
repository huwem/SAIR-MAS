"""配置序列化模块。

支持 FrameworkConfig 的 YAML/JSON 导入导出、实验配置保存与加载、
预设导出为独立配置文件。
"""

import json
from pathlib import Path
from typing import Optional

from mas_topo.config.schema import FrameworkConfig
from mas_topo.config.loader import _substitute_env_vars


def save_config(config: FrameworkConfig, path: str, format: Optional[str] = None) -> None:
    """将 FrameworkConfig 保存到文件。

    Args:
        config: 配置对象。
        path: 文件路径。
        format: 格式，"yaml" / "json" / None（自动根据后缀推断）。
    """
    p = Path(path)
    fmt = format or _infer_format(p.suffix)

    data = config.to_dict()

    p.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "yaml":
        try:
            import yaml
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        except ImportError:
            raise ImportError("PyYAML required for YAML export: pip install pyyaml")
    elif fmt == "json":
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}. Use 'yaml' or 'json'.")


def load_config(path: str) -> FrameworkConfig:
    """从 YAML 或 JSON 文件加载配置（带环境变量替换）。"""
    from mas_topo.config.loader import load_config as _loader
    return _loader(path)


def config_to_yaml_string(config: FrameworkConfig) -> str:
    """将配置导出为 YAML 字符串。"""
    try:
        import yaml
        return yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True)
    except ImportError:
        raise ImportError("PyYAML required: pip install pyyaml")


def config_to_json_string(config: FrameworkConfig, indent: int = 2) -> str:
    """将配置导出为 JSON 字符串。"""
    return json.dumps(config.to_dict(), indent=indent, ensure_ascii=False)


def _infer_format(suffix: str) -> str:
    if suffix in (".yaml", ".yml"):
        return "yaml"
    if suffix == ".json":
        return "json"
    raise ValueError(f"Cannot infer format from suffix '{suffix}'")
