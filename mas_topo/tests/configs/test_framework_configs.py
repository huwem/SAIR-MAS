"""验证关键框架/实验配置可正确加载并构建 Graph。"""

import pytest

from mas_topo.config.loader import load_config
from mas_topo.graph import Graph

# 当前维护的活跃配置清单（相对项目根目录）
CONFIG_PATHS = [
    # --- Code tasks ---
    "configs/experiments/dytopo_code_local.yaml",
    "configs/experiments/p2p_free_code_local.yaml",
    "configs/experiments/sair_code_local.yaml",
    "configs/experiments/vanilla_code_local.yaml",
    # --- Reasoning tasks ---
    "configs/experiments/dytopo_reasoning_local.yaml",
    "configs/experiments/p2p_free_reasoning_local.yaml",
    "configs/experiments/sair_reasoning_local.yaml",
    "configs/experiments/vanilla_reasoning_local.yaml",
]


@pytest.mark.parametrize("config_path", CONFIG_PATHS)
def test_framework_config_builds_graph(config_path: str, project_root):
    """每个配置都应能加载并构建出合法的 Graph。"""
    path = project_root / config_path
    if not path.exists():
        pytest.skip(f"Config not found: {config_path}")

    config = load_config(str(path))
    framework_config = getattr(config, "framework", config)
    graph = Graph.from_config(framework_config)
    assert graph.agents
    assert graph.routing_strategy is not None
    assert graph.decision_strategy is not None
