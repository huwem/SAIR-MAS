"""配置模块。

提供配置数据类、加载器、发现和序列化。
"""

from mas_topo.config.schema import (
    AgentConfig, TopologyConfig, DecisionConfig,
    RoutingConfig, LLMConfig, FrameworkConfig,
)
from mas_topo.config.loader import load_config, load_config_from_dict
from mas_topo.config.discovery import list_presets, get_preset_metadata

__all__ = [
    "AgentConfig", "TopologyConfig", "DecisionConfig",
    "RoutingConfig", "LLMConfig", "FrameworkConfig",
    "load_config", "load_config_from_dict",
    "list_presets", "get_preset_metadata",
]
