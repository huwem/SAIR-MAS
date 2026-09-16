"""全局注册表实例。

此模块不导入任何 mas_topo 子模块，避免循环导入。
各策略模块从此处导入注册表实例并使用 @register 装饰器。
"""

from mas_topo.registry import Registry

agent_registry = Registry("agent")
llm_registry = Registry("llm")
topology_registry = Registry("topology")
decision_registry = Registry("decision")
routing_registry = Registry("routing")
prompt_registry = Registry("prompt")
tool_registry = Registry("tool")
dataset_registry = Registry("dataset")
evaluator_registry = Registry("evaluator")
memory_registry = Registry("memory")
