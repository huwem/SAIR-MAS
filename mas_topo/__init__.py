"""
mas-topo: LLM 多智能体拓扑/路由/通信实验框架。

提供可插拔的拓扑策略、路由策略和决策策略，
支持通过配置文件或代码快速搭建多智能体实验环境。

Quick Start:
    from mas_topo import Graph, load_config

    config = load_config("configs/dytopo.yaml")
    graph = Graph.from_config(config)
    result = graph.run("Write a Python function to check palindromes.")
    print(result.final_answer)
"""

from mas_topo._registries import (
    agent_registry,
    llm_registry,
    topology_registry,
    decision_registry,
    routing_registry,
    prompt_registry,
    tool_registry,
    dataset_registry,
    evaluator_registry,
    memory_registry,
)

from mas_topo.registry import Registry
from mas_topo.message import Message, AgentOutput
from mas_topo.context import ExecutionContext
from mas_topo.graph import Graph, GraphResult, RoundResult

from mas_topo.agent import BaseAgent, LLMAgent
from mas_topo.llm import LLMAdapter, OpenAICompatAdapter, LLMResponse, LLMUsage

from mas_topo.topology.base import TopologyStrategy
from mas_topo.topology.fixed import FixedTopology
from mas_topo.topology.semantic import SemanticMatchingTopology
from mas_topo.topology.cosine import CosineTopology
from mas_topo.topology.manual import ManualTopology
from mas_topo.topology.neighbor import NeighborTopology

from mas_topo.routing.base import RoutingStrategy
from mas_topo.routing.graph import GraphRouting
from mas_topo.routing.semantic import SemanticRouting
from mas_topo.routing.broadcast import BroadcastRouting

from mas_topo.decision.base import DecisionStrategy, DecisionResult
from mas_topo.decision.manager import ManagerDecision
from mas_topo.decision.voting import VotingDecision
from mas_topo.decision.direct import DirectDecision
from mas_topo.decision.llm_judge import LLMJudgeDecision

from mas_topo.config.schema import (
    AgentConfig, TopologyConfig, DecisionConfig,
    RoutingConfig, LLMConfig, FrameworkConfig,
)
from mas_topo.config.experiment_schema import (
    ExperimentConfig, DatasetConfig, EvaluationConfig, OutputConfig,
)
from mas_topo.config.loader import load_config, load_config_from_dict
from mas_topo.config.discovery import (
    list_presets,
    get_preset_metadata,
    list_registered_strategies,
    get_strategy_metadata,
    get_framework_json_schema,
    validate_config,
    preview_topology,
    build_framework_config,
)
from mas_topo.config.serializer import (
    save_config,
    config_to_yaml_string,
    config_to_json_string,
)

from mas_topo.datasets.base import BaseDatasetLoader, DatasetSample
from mas_topo.evaluation.base import Evaluator, EvaluationResult
from mas_topo.tools.base import Tool
from mas_topo.tools.builtin import extract_python_code

from mas_topo.prompt.base import PromptProvider
from mas_topo.prompt.template import TemplatePromptProvider

from mas_topo.experiment.runner import ExperimentRunner
from mas_topo.experiment.tracker import ExperimentTracker, RunMetrics, RoundMetrics
from mas_topo.experiment.topology_logger import TopologyLogger
from mas_topo.experiment.dataset_runner import DatasetExperimentRunner, DatasetExperimentSummary

from mas_topo.datasets.humaneval_loader import HumanEvalLoader, HumanEvalSample
from mas_topo.datasets.aime_loader import AIMELoader, AIMESample
from mas_topo.datasets.math_loader import MATHLoader, MATHSample


# ── 注册 Agent 和 LLM 到全局注册表 ─────────────────────────

@agent_registry.register("LLMAgent")
class _LLMAgentRegistered(LLMAgent):
    """注册到全局注册表的 LLMAgent。"""
    def __init__(self, name="", role="", llm_adapter=None,
                 system_prompt=None, output_format="text",
                 output_schema=None, tools=None, memory_limit=500, **kwargs):
        super().__init__(name=name, role=role, llm_adapter=llm_adapter,
                         system_prompt=system_prompt, output_format=output_format,
                         output_schema=output_schema, tools=tools,
                         memory_limit=memory_limit, **kwargs)


@llm_registry.register("openai_compat")
class _OpenAICompatRegistered(OpenAICompatAdapter):
    """注册到全局注册表的 OpenAI 兼容适配器。"""
    def __init__(self, model="gpt-4o", api_key=None, base_url=None,
                 temperature=0.2, max_tokens=1000, timeout=120.0,
                 request_timeout=900.0, json_mode=False, disable_thinking=False,
                 thinking_budget=0, stream=False,
                 **kwargs):
        super().__init__(model=model, api_key=api_key, base_url=base_url,
                         temperature=temperature, max_tokens=max_tokens,
                         timeout=timeout, request_timeout=request_timeout,
                         json_mode=json_mode,
                         disable_thinking=disable_thinking,
                         thinking_budget=thinking_budget, stream=stream)


# ── 触发所有策略模块的 @register 装饰器 ─────────────────────

def _ensure_registrations():
    """确保所有策略模块已导入，触发 @register 装饰器。"""
    Graph._ensure_registrations()


__all__ = [
    # Registries
    "agent_registry", "llm_registry", "topology_registry",
    "decision_registry", "routing_registry", "prompt_registry", "tool_registry",
    "dataset_registry", "evaluator_registry", "memory_registry",
    "Registry",
    # Core
    "Message", "AgentOutput", "ExecutionContext",
    "Graph", "GraphResult", "RoundResult",
    # Agent
    "BaseAgent", "LLMAgent",
    # LLM
    "LLMAdapter", "OpenAICompatAdapter", "LLMResponse", "LLMUsage",
    # Topology
    "TopologyStrategy",
    "FixedTopology", "SemanticMatchingTopology",
    "CosineTopology", "ManualTopology", "NeighborTopology",
    # Routing
    "RoutingStrategy", "GraphRouting", "SemanticRouting", "BroadcastRouting",
    # Decision
    "DecisionStrategy", "DecisionResult",
    "ManagerDecision", "VotingDecision", "DirectDecision", "LLMJudgeDecision",
    # Config
    "AgentConfig", "TopologyConfig", "DecisionConfig",
    "RoutingConfig", "LLMConfig", "FrameworkConfig",
    "ExperimentConfig", "DatasetConfig", "EvaluationConfig", "OutputConfig",
    "load_config", "load_config_from_dict",
    "list_presets", "get_preset_metadata",
    "save_config",
    "config_to_yaml_string", "config_to_json_string",
    "list_registered_strategies", "get_strategy_metadata",
    "get_framework_json_schema", "validate_config",
    "preview_topology", "build_framework_config",
    # Tools
    "Tool", "extract_python_code",
    # Prompt
    "PromptProvider", "TemplatePromptProvider",
    # Experiment
    "ExperimentRunner", "ExperimentTracker", "RunMetrics", "RoundMetrics",
    "TopologyLogger",
    "DatasetExperimentRunner", "DatasetExperimentSummary",
    # Datasets
    "BaseDatasetLoader", "DatasetSample",
    "HumanEvalLoader", "HumanEvalSample",
    "AIMELoader", "AIMESample",
    "MATHLoader", "MATHSample",
    # Evaluation
    "Evaluator", "EvaluationResult",
]
