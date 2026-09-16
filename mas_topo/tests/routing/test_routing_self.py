import numpy as np
from mas_topo.agent import LLMAgent
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput, Channel
from mas_topo.routing.self_routing import SelfRouting


def test_self_routing_delivers_private_messages_and_logs_ledger():
    router = SelfRouting(neighbor_policy="strict")
    adj = np.array([[0, 1], [1, 0]])
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["B"], "illocution": "Directive", "content": "do X"}
            ]
        }),
        "B": AgentOutput(),
    }
    context = ExecutionContext(round_num=1)
    messages, ledger_entries = router.route_and_log(["A", "B"], outputs, adj, context)
    assert len(messages["B"]) == 1
    assert messages["B"][0].sender_id == "A"
    assert messages["B"][0].channel == Channel.FT_PRIVATE
    assert len(ledger_entries) == 1
    assert ledger_entries[0].illocution == "Directive"
    assert "B" in ledger_entries[0].recipients
    # Ledger is attached to context
    assert context.ledger is not None


def test_self_routing_stores_answer_in_ledger_not_message():
    router = SelfRouting(neighbor_policy="strict")
    adj = np.array([[0, 1], [1, 0]])
    outputs = {
        "A": AgentOutput(
            answer="def foo(): return 42",
            metadata={
                "private_routing": [
                    {"recipients": ["B"], "illocution": "Assertive", "content": "Here is the code"}
                ]
            }
        ),
        "B": AgentOutput(),
    }
    context = ExecutionContext(round_num=1)
    messages, ledger_entries = router.route_and_log(["A", "B"], outputs, adj, context)

    # 消息内容不再包含 answer
    assert len(messages["B"]) == 1
    assert "def foo(): return 42" not in messages["B"][0].content
    assert "[Answer from A]" not in messages["B"][0].content

    # ledger 中保存 answer_snapshot
    assert len(ledger_entries) == 1
    assert ledger_entries[0].answer_snapshot == "def foo(): return 42"


def test_self_routing_blocks_non_neighbor():
    router = SelfRouting(neighbor_policy="strict")
    adj = np.array([[0, 0], [0, 0]])
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["B"], "illocution": "Directive", "content": "do X"}
            ]
        }),
        "B": AgentOutput(),
    }
    context = ExecutionContext(round_num=1)
    messages, ledger_entries = router.route_and_log(["A", "B"], outputs, adj, context)
    assert len(messages["B"]) == 0
    assert len(ledger_entries) == 0


def test_self_routing_broadcast_to_all():
    router = SelfRouting(neighbor_policy="strict")
    adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["All"], "illocution": "Assertive", "content": "hello"}
            ]
        }),
        "B": AgentOutput(),
        "C": AgentOutput(),
    }
    context = ExecutionContext(round_num=2)
    messages, ledger_entries = router.route_and_log(["A", "B", "C"], outputs, adj, context)
    assert len(messages["B"]) == 1
    assert len(messages["C"]) == 1
    assert messages["B"][0].channel == Channel.BROADCAST
    assert len(ledger_entries) == 1
    assert set(ledger_entries[0].recipients) == {"B", "C"}


def test_self_routing_no_relay_by_default():
    # A -> C 不是邻居，严格单跳策略下应丢弃
    router = SelfRouting(neighbor_policy="strict")
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # path A-B-C
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["C"], "illocution": "Directive", "content": "do X"}
            ]
        }),
        "B": AgentOutput(),
        "C": AgentOutput(),
    }
    context = ExecutionContext(round_num=1)
    messages, ledger_entries = router.route_and_log(["A", "B", "C"], outputs, adj, context)
    assert len(messages["C"]) == 0
    assert len(ledger_entries) == 0


def test_self_routing_agent_forward_by_directive():
    # Agent 通过标准 Directive 语旨请邻居 B 代为转发给 C；框架只负责单跳投递。
    router = SelfRouting(neighbor_policy="strict")
    adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])  # path A-B-C
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {
                    "recipients": ["B"],
                    "illocution": "Directive",
                    "content": "Please forward to C: [Directive] do X",
                }
            ]
        }),
        "B": AgentOutput(),
        "C": AgentOutput(),
    }
    context = ExecutionContext(round_num=1)
    messages, ledger_entries = router.route_and_log(["A", "B", "C"], outputs, adj, context)

    # B 应收到 A 发来的 Directive 转发请求
    assert len(messages["B"]) == 1
    forward_msg = messages["B"][0]
    assert forward_msg.sender_id == "A"
    assert forward_msg.metadata.get("illocution_category") == "Directive"
    assert "forward to C" in forward_msg.content

    # ledger 中应记录该 Directive 行为（不是非标准 Relay）
    assert len(ledger_entries) == 1
    assert ledger_entries[0].illocution == "Directive"
    assert ledger_entries[0].sender == "A"
    assert ledger_entries[0].recipients == ["B"]


def test_self_routing_dynamic_neighbor_update_by_topology():
    # 局部邻居表由外部拓扑（Graph/Topology 策略）维护，而非 Agent 主动 CONNECT。
    # 模拟拓扑从断开变为连接：A 在表中的自身条目被外部更新为 {B} 后，A 即可发送。
    router = SelfRouting(neighbor_policy="strict")
    adj_disconnected = np.array([[0, 0], [0, 0]])
    adj_connected = np.array([[0, 1], [1, 0]])
    agent_a = LLMAgent(name="A")
    agent_b = LLMAgent(name="B")
    agents = [agent_a, agent_b]

    context = ExecutionContext(round_num=1)
    context.agents = agents

    # Round 1: A 的自身条目为空，尝试发给 B 失败
    agent_a.local_neighbor_table = {"A": set()}
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["B"], "illocution": "Directive", "content": "do X"}
            ]
        }),
        "B": AgentOutput(),
    }
    messages, _ = router.route_and_log(["A", "B"], outputs, adj_disconnected, context)
    assert len(messages["B"]) == 0

    # Round 2: 外部拓扑变化，Graph 会同步更新 A.local_neighbor_table["A"]；这里手动模拟该同步
    agent_a.local_neighbor_table = {"A": {"B"}}
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["B"], "illocution": "Directive", "content": "do X"}
            ]
        }),
        "B": AgentOutput(),
    }
    messages, _ = router.route_and_log(["A", "B"], outputs, adj_connected, context)
    assert len(messages["B"]) == 1
    assert messages["B"][0].sender_id == "A"


def test_self_routing_dynamic_neighbor_remove_by_topology():
    # 模拟外部拓扑从连接变为断开：A.local_neighbor_table["A"] 被外部更新为空后，A 无法发送给 B。
    router = SelfRouting(neighbor_policy="strict")
    adj_connected = np.array([[0, 1], [1, 0]])
    adj_disconnected = np.array([[0, 0], [0, 0]])
    agent_a = LLMAgent(name="A", local_neighbor_table={"A": {"B"}})
    agent_b = LLMAgent(name="B", local_neighbor_table={"B": {"A"}})
    agents = [agent_a, agent_b]

    context = ExecutionContext(round_num=1)
    context.agents = agents

    # Round 1: A 可以发给 B
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["B"], "illocution": "Directive", "content": "do X"}
            ]
        }),
        "B": AgentOutput(),
    }
    messages, _ = router.route_and_log(["A", "B"], outputs, adj_connected, context)
    assert len(messages["B"]) == 1

    # Round 2: 外部拓扑断开，Graph 同步清空 A.local_neighbor_table["A"]
    agent_a.local_neighbor_table = {"A": set()}
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["B"], "illocution": "Directive", "content": "do X"}
            ]
        }),
        "B": AgentOutput(),
    }
    messages, _ = router.route_and_log(["A", "B"], outputs, adj_disconnected, context)
    assert len(messages["B"]) == 0


def test_self_routing_agent_forward_chain_by_directive():
    # 模拟两跳社会性转发：A -> B -> C -> D，全部使用标准 Directive 语旨。
    router = SelfRouting(neighbor_policy="strict")
    adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ])

    # Round 1: A -> B (请 B 转发给 D)
    outputs_r1 = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["B"], "illocution": "Directive", "content": "Please forward to D: [Directive] do X"}
            ]
        }),
    }
    for name in ["B", "C", "D"]:
        outputs_r1[name] = AgentOutput()
    context = ExecutionContext(round_num=1)
    messages_r1, _ = router.route_and_log(["A", "B", "C", "D"], outputs_r1, adj, context)
    assert len(messages_r1["B"]) == 1

    # Round 2: B -> C (请 C 转发给 D)
    outputs_r2 = {
        "B": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["C"], "illocution": "Directive", "content": "Please forward to D: [Directive] do X"}
            ]
        }),
    }
    for name in ["A", "C", "D"]:
        outputs_r2[name] = AgentOutput()
    context = ExecutionContext(round_num=2)
    messages_r2, _ = router.route_and_log(["A", "B", "C", "D"], outputs_r2, adj, context)
    assert len(messages_r2["C"]) == 1

    # Round 3: C -> D (Directive)
    outputs_r3 = {
        "C": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["D"], "illocution": "Directive", "content": "do X"}
            ]
        }),
    }
    for name in ["A", "B", "D"]:
        outputs_r3[name] = AgentOutput()
    context = ExecutionContext(round_num=3)
    messages_r3, _ = router.route_and_log(["A", "B", "C", "D"], outputs_r3, adj, context)
    assert len(messages_r3["D"]) == 1
    assert messages_r3["D"][0].metadata.get("illocution_category") == "Directive"


def test_update_topology_view_tool_updates_agent_view():
    # Agent 调用 update_topology_view 工具更新统一局部邻居表中关于其他 Agent 的条目。
    from mas_topo.tools.builtin import UpdateTopologyViewTool

    agent = LLMAgent(name="A", local_neighbor_table={"A": {"B"}})
    tool = UpdateTopologyViewTool()
    result = tool.run(
        agent=agent,
        topology_info={"B": ["A", "C"], "C": ["B", "D"]},
    )
    assert result["success"] is True
    assert agent.local_neighbor_table["B"] == {"A", "C"}
    assert agent.local_neighbor_table["C"] == {"B", "D"}
    # 工具不应覆盖框架维护的 agent 自身邻居条目
    assert agent.local_neighbor_table["A"] == {"B"}


def test_update_topology_view_tool_protects_self_entry():
    # Agent 不能通过 update_topology_view 工具修改自身的物理一跳邻居条目。
    from mas_topo.tools.builtin import UpdateTopologyViewTool

    agent = LLMAgent(name="A", local_neighbor_table={"A": {"B"}})
    tool = UpdateTopologyViewTool()
    result = tool.run(
        agent=agent,
        topology_info={"A": ["B", "C", "D"]},
    )
    assert result["success"] is True
    # 自身条目保持原样
    assert agent.local_neighbor_table["A"] == {"B"}


def test_agent_run_tool_update_topology_view():
    # LLMAgent._run_tool 自动注入 agent 上下文并执行 update_topology_view。
    agent = LLMAgent(name="A", local_neighbor_table={"A": {"B"}})
    tool_result = agent._run_tool(
        "update_topology_view",
        {"topology_info": {"B": ["A", "C"]}},
    )
    assert tool_result["success"] is True
    assert agent.local_neighbor_table["B"] == {"A", "C"}


def test_self_routing_resolves_recipient_with_role_annotation():
    # LLM 可能从 topology view 中学到 "Tester (Code Tester)" 这种格式，
    # 解析时应自动去掉角色说明，只按 Agent 名称匹配。
    router = SelfRouting(neighbor_policy="strict")
    adj = np.array([[0, 1], [1, 0]])
    outputs = {
        "A": AgentOutput(metadata={
            "private_routing": [
                {"recipients": ["B (Role B)"], "illocution": "Directive", "content": "do X"},
                {"recipients": ["B - Role B"], "illocution": "Assertive", "content": "do Y"},
                {"recipients": ["unknown (Role)"], "illocution": "Assertive", "content": "do Z"},
            ]
        }),
        "B": AgentOutput(),
    }
    context = ExecutionContext(round_num=1)
    messages, ledger_entries = router.route_and_log(["A", "B"], outputs, adj, context)
    assert len(messages["B"]) == 2
    assert messages["B"][0].metadata["illocution_category"] == "Directive"
    assert messages["B"][1].metadata["illocution_category"] == "Assertive"
    # 未知接收者被丢弃，不生成 ledger 条目
    assert len(ledger_entries) == 2
