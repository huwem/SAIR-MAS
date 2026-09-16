"""组件发现与配置构建模块。

为图形化界面提供策略发现、参数描述、配置验证和可视化构建能力。
GUI 可通过此模块：
- 发现所有可用的策略及其元数据
- 获取每个策略的可配置参数字段
- 动态构建并验证 FrameworkConfig
- 预览配置效果（如 Agent 拓扑图结构）
"""

from pathlib import Path
from typing import Optional

from mas_topo.config.schema import (
    FrameworkConfig,
    AgentConfig,
    TopologyConfig,
    RoutingConfig,
    DecisionConfig,
    LLMConfig,
)
from mas_topo.config.experiment_schema import ExperimentConfig

# 配置文件目录（相对于项目根目录）
_CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


# ── 策略发现 ──────────────────────────────────────────

def list_registered_strategies(registry_name: str) -> list[dict]:
    """列出某注册表中所有已注册策略及其元数据。

    Args:
        registry_name: 注册表名，如 "topology", "routing", "decision", "agent", "llm".

    Returns:
        每个元素为 {"name": str, "display_name": str, "description": str,
                   "config_schema": dict, "category": str}。
    """
    from mas_topo import (
        agent_registry, llm_registry,
        topology_registry, decision_registry, routing_registry,
        tool_registry,
    )
    reg_map = {
        "agent": agent_registry,
        "llm": llm_registry,
        "topology": topology_registry,
        "decision": decision_registry,
        "routing": routing_registry,
        "tool": tool_registry,
    }
    if registry_name not in reg_map:
        raise ValueError(f"Unknown registry '{registry_name}'. Available: {list(reg_map.keys())}")
    return reg_map[registry_name].list_with_metadata()


def get_strategy_metadata(registry_name: str, strategy_name: str) -> dict:
    """获取单个策略的元数据。"""
    from mas_topo import (
        agent_registry, llm_registry,
        topology_registry, decision_registry, routing_registry,
    )
    reg_map = {
        "agent": agent_registry,
        "llm": llm_registry,
        "topology": topology_registry,
        "decision": decision_registry,
        "routing": routing_registry,
    }
    reg = reg_map.get(registry_name)
    if reg is None:
        raise ValueError(f"Unknown registry '{registry_name}'")
    return reg.metadata(strategy_name)


def get_framework_json_schema() -> dict:
    """获取 FrameworkConfig 的完整 JSON Schema，供 GUI 自动生成配置表单。"""
    return FrameworkConfig.model_json_schema()


# ── 配置文件发现 ──────────────────────────────────────

def list_presets() -> list[str]:
    """列出 configs/ 目录下所有可用的配置文件（不含扩展名）。"""
    if not _CONFIGS_DIR.exists():
        return []
    names = []
    for f in sorted(_CONFIGS_DIR.iterdir()):
        if f.suffix in (".yaml", ".yml", ".json"):
            names.append(f.stem)
    return names


def get_preset_metadata(name: str) -> dict:
    """获取配置文件的元数据（名称、描述、Agent、策略等）。"""
    from mas_topo.config.loader import load_config

    for ext in (".yaml", ".yml", ".json"):
        path = _CONFIGS_DIR / f"{name}{ext}"
        if path.exists():
            config = load_config(str(path))
            # 兼容 ExperimentConfig 和 FrameworkConfig
            if isinstance(config, ExperimentConfig):
                fw = config.framework
                return {
                    "name": name,
                    "path": str(path),
                    "description": config.description or fw.description or "",
                    "experiment_name": config.name or "",
                    "dataset": config.dataset.loader,
                    "evaluation": config.evaluation.metric,
                    "agents": [a.name for a in fw.agents],
                    "topology": fw.topology.strategy,
                    "routing": fw.routing.strategy,
                    "decision": fw.decision.strategy,
                    "max_rounds": fw.max_rounds,
                }
            else:
                return {
                    "name": name,
                    "path": str(path),
                    "description": config.description or "",
                    "agents": [a.name for a in config.agents],
                    "topology": config.topology.strategy,
                    "routing": config.routing.strategy,
                    "decision": config.decision.strategy,
                    "max_rounds": config.max_rounds,
                }

    raise KeyError(f"Config file not found for '{name}'. Available: {list_presets()}")


# ── 配置验证 ──────────────────────────────────────────

def validate_config(data: dict) -> tuple[bool, list[str]]:
    """验证配置字典是否有效。

    自动检测配置类型（ExperimentConfig 或 FrameworkConfig）并验证。

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # 自动检测配置类型
    is_experiment = "experiment" in data or ("framework" in data and ("dataset" in data or "evaluation" in data))

    try:
        if is_experiment:
            ExperimentConfig.model_validate(data if "framework" in data else data.get("experiment", data))
        else:
            FrameworkConfig.model_validate(data)
    except Exception as e:
        errors.append(str(e))
        return False, errors
    return True, []


# ── 可视化构建辅助 ──────────────────────────────────────

def preview_topology(config: FrameworkConfig) -> dict:
    """根据拓扑策略配置预览拓扑结构。

    返回节点列表和边列表，供 GUI 渲染拓扑图。
    """
    nodes = []
    for i, a in enumerate(config.agents):
        nodes.append({
            "id": a.name or f"Agent{i}",
            "role": a.role or a.type,
            "type": a.type,
        })

    edges = []
    strategy = config.topology.strategy

    if strategy == "fixed" and config.topology.pattern:
        edges = _preview_fixed_topology(nodes, config.topology.pattern)
    elif strategy in ("semantic", "cosine"):
        edges = _preview_dynamic_topology(nodes, strategy)
    elif strategy == "manual" and config.topology.adjacency:
        adj = config.topology.adjacency
        for i in range(len(adj)):
            for j in range(len(adj[i])):
                if adj[i][j] == 1:
                    edges.append({"source": nodes[i]["id"], "target": nodes[j]["id"]})
    else:
        edges = _preview_fixed_topology(nodes, "full_connected")

    return {"nodes": nodes, "edges": edges, "strategy": strategy}


def _preview_fixed_topology(nodes: list[dict], pattern: str) -> list[dict]:
    """基于固定模式生成预览边。"""
    N = len(nodes)
    if N == 0:
        return []

    edges = []
    if pattern == "full_connected":
        for i in range(N):
            for j in range(N):
                if i != j:
                    edges.append({"source": nodes[i]["id"], "target": nodes[j]["id"]})
    elif pattern == "chain":
        for i in range(N - 1):
            edges.append({"source": nodes[i]["id"], "target": nodes[i + 1]["id"]})
    elif pattern == "star" and N >= 2:
        for i in range(1, N):
            edges.append({"source": nodes[0]["id"], "target": nodes[i]["id"]})
            edges.append({"source": nodes[i]["id"], "target": nodes[0]["id"]})
    elif pattern == "ring":
        for i in range(N):
            edges.append({"source": nodes[i]["id"], "target": nodes[(i + 1) % N]["id"]})
    elif pattern == "tree":
        for i in range(1, N):
            parent = (i - 1) // 2
            edges.append({"source": nodes[parent]["id"], "target": nodes[i]["id"]})
    elif pattern == "mesh":
        import math
        sqrt_n = int(math.sqrt(N))
        if sqrt_n * sqrt_n == N and N > 4:
            for i in range(N):
                if (i + 1) % sqrt_n != 0:
                    edges.append({"source": nodes[i]["id"], "target": nodes[i + 1]["id"]})
                    edges.append({"source": nodes[i + 1]["id"], "target": nodes[i]["id"]})
                if i < N - sqrt_n:
                    edges.append({"source": nodes[i]["id"], "target": nodes[i + sqrt_n]["id"]})
                    edges.append({"source": nodes[i + sqrt_n]["id"], "target": nodes[i]["id"]})
    return edges


def _preview_dynamic_topology(nodes: list[dict], strategy: str) -> list[dict]:
    """动态拓扑预览：显示为全连接，标注策略名。"""
    N = len(nodes)
    edges = []
    for i in range(N):
        for j in range(N):
            if i != j:
                edges.append({
                    "source": nodes[i]["id"],
                    "target": nodes[j]["id"],
                    "dynamic": True,
                    "strategy": strategy,
                })
    return edges


# ── 动态配置构建 ────────────────────────────────────────

def build_experiment_config(
    agents: list[dict],
    dataset: Optional[dict] = None,
    evaluation: Optional[dict] = None,
    output: Optional[dict] = None,
    topology: Optional[dict] = None,
    routing: Optional[dict] = None,
    decision: Optional[dict] = None,
    llm: Optional[dict] = None,
    **kwargs,
) -> ExperimentConfig:
    """从组件字典动态构建 ExperimentConfig。

    Args:
        agents: Agent 配置字典列表。
        dataset: DatasetConfig 字段字典。
        evaluation: EvaluationConfig 字段字典。
        output: OutputConfig 字段字典。
        topology: TopologyConfig 字段字典。
        routing: RoutingConfig 字段字典。
        decision: DecisionConfig 字段字典。
        llm: LLMConfig 字段字典。
        **kwargs: ExperimentConfig 的其他字段（name, description, max_rounds 等）。

    Returns:
        ExperimentConfig 实例。
    """
    from mas_topo.config.experiment_schema import DatasetConfig, EvaluationConfig, OutputConfig

    agent_configs = [AgentConfig.model_validate(a) for a in agents]
    topo_config = TopologyConfig.model_validate(topology or {})
    rout_config = RoutingConfig.model_validate(routing or {})
    dec_config = DecisionConfig.model_validate(decision or {})
    llm_config = LLMConfig.model_validate(llm or {})
    dataset_config = DatasetConfig.model_validate(dataset or {})
    eval_config = EvaluationConfig.model_validate(evaluation or {})
    output_config = OutputConfig.model_validate(output or {})

    framework = FrameworkConfig(
        agents=agent_configs,
        topology=topo_config,
        routing=rout_config,
        decision=dec_config,
        llm=llm_config,
        max_rounds=kwargs.pop("max_rounds", 10),
        verbose=kwargs.pop("verbose", True),
        log_dir=kwargs.pop("log_dir", None),
        description=kwargs.pop("description", None),
    )

    return ExperimentConfig(
        framework=framework,
        dataset=dataset_config,
        evaluation=eval_config,
        output=output_config,
        **kwargs,
    )


def build_framework_config(
    agents: list[dict],
    topology: Optional[dict] = None,
    routing: Optional[dict] = None,
    decision: Optional[dict] = None,
    llm: Optional[dict] = None,
    **kwargs,
) -> FrameworkConfig:
    """从组件字典动态构建 FrameworkConfig。

    Args:
        agents: Agent 配置字典列表，每个字典对应 AgentConfig 字段。
        topology: TopologyConfig 字段字典。
        routing: RoutingConfig 字段字典。
        decision: DecisionConfig 字段字典。
        llm: LLMConfig 字段字典。
        **kwargs: FrameworkConfig 的其他字段（如 max_rounds, verbose, log_dir）。

    Example:
        config = build_framework_config(
            agents=[
                {"name": "Manager", "role": "Coordinator"},
                {"name": "Worker", "role": "Implementer"},
            ],
            topology={"strategy": "semantic", "threshold": 0.3},
            llm={"model": "gpt-4o"},
        )
    """
    agent_configs = [AgentConfig.model_validate(a) for a in agents]
    topo_config = TopologyConfig.model_validate(topology or {})
    rout_config = RoutingConfig.model_validate(routing or {})
    dec_config = DecisionConfig.model_validate(decision or {})
    llm_config = LLMConfig.model_validate(llm or {})

    return FrameworkConfig(
        agents=agent_configs,
        topology=topo_config,
        routing=rout_config,
        decision=dec_config,
        llm=llm_config,
        **kwargs,
    )
