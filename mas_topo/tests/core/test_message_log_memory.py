"""消息日志记忆更新策略测试。"""

import pytest

from mas_topo._registries import memory_registry
from mas_topo.config.loader import load_config
from mas_topo.graph import Graph
from mas_topo.memory.ledger_update import MessageLogMemoryUpdate
from mas_topo.message import AgentOutput, Channel, Message


class FakeAgent:
    def __init__(self, name):
        self.name = name
        self.memory = []

    def append_memory(self, content, round_num=0):
        self.memory.append(content)


def test_registry_maps_ledger_and_message_log_to_distinct_classes():
    """ledger 与 message_log 必须注册到不同的类，避免同名冲突。"""
    ledger_cls = memory_registry.get_class("ledger")
    message_log_cls = memory_registry.get_class("message_log")
    assert ledger_cls is not None
    assert message_log_cls is not None
    assert ledger_cls is not message_log_cls
    assert ledger_cls.__name__ == "LedgerMemoryUpdate"
    assert message_log_cls.__name__ == "MessageLogMemoryUpdate"


def test_message_log_memory_update_accepts_context():
    """graph.py 调用 update 时总传入 context，message_log 策略必须接受。"""
    strategy = MessageLogMemoryUpdate()
    agent = FakeAgent("Developer")
    # 不应抛出 TypeError
    strategy.update(
        agents=[agent],
        agent_names=["Developer"],
        routed_messages={},
        agent_outputs={},
        round_num=1,
        aggregation_order=[0],
        context=None,
    )


def test_message_log_memory_records_own_private_routing():
    """自身 private_routing 应以摘要形式写入记忆。"""
    strategy = MessageLogMemoryUpdate()
    agent = FakeAgent("Developer")
    strategy.update(
        agents=[agent],
        agent_names=["Developer", "Tester"],
        routed_messages={},
        agent_outputs={
            "Developer": AgentOutput(
                private_routing=[
                    {"recipients": ["Tester"], "illocution": "Assertive", "content": "code done"}
                ],
            )
        },
        round_num=1,
    )
    assert any("Assertive -> Tester" in m for m in agent.memory)
    assert any("code done" in m for m in agent.memory)


def test_message_log_memory_records_received_messages():
    """收到的 private/broadcast 消息应写入记忆。"""
    strategy = MessageLogMemoryUpdate()
    agent = FakeAgent("Developer")
    strategy.update(
        agents=[agent],
        agent_names=["Tester", "Developer"],
        routed_messages={
            "Developer": [
                Message(
                    content="please fix bug",
                    sender_id="Tester",
                    receiver_id="Developer",
                    channel=Channel.FT_PRIVATE,
                )
            ]
        },
        agent_outputs={},
        round_num=1,
    )
    assert any("please fix bug" in m for m in agent.memory)
    assert any("From Tester" in m for m in agent.memory)


@pytest.mark.parametrize("config_path, expected_memory", [
    ("configs/experiments/vanilla_code_local.yaml", "MessageLogMemoryUpdate"),
    ("configs/experiments/dytopo_code_local.yaml", "DefaultMemoryUpdate"),
])
def test_code_local_configs_build_graph(config_path, expected_memory):
    """验证配置能成功构建 Graph，不会在运行时暴露接口不匹配。"""
    config = load_config(config_path)
    framework_config = getattr(config, "framework", config)
    graph = Graph.from_config(framework_config)
    assert graph.memory_strategy.__class__.__name__ == expected_memory
