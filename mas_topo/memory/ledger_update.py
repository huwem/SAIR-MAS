"""Ledger Memory Update — 全局契约账本记忆更新策略。

纯语旨架构下的记忆管理：
- 所有 Agent 统一通过 private_routing 通信，不再依赖 public_content。
- Workers 仅接收发给自己的 private 消息和 "All" 广播，保持局部视角。
- Manager（通过 role 前缀 "Workflow Orchestrator" 识别）额外接收全局契约账本
  摘要，具备全景审计能力。
- 自身输出以 private_routing 摘要形式写入记忆。

通过注册表扩展，不影响 DefaultMemoryUpdate。
"""

from typing import Optional

from mas_topo._registries import memory_registry
from mas_topo.memory.base import MemoryUpdateStrategy
from mas_topo.message import AgentOutput, Message


@memory_registry.register(
    "message_log",
    display_name="消息日志记忆更新",
    description="追加自身输出（private_routing/answer）与收到的 private/broadcast 消息；Manager 额外注入一轮全局消息摘要。",
)
class MessageLogMemoryUpdate(MemoryUpdateStrategy):
    """消息日志记忆更新策略。

    支持通道：semantic-private, spatial, ft-private, ft-broadcast
    """

    SUPPORTED_CHANNELS = ("semantic-private", "spatial", "ft-private", "ft-broadcast")

    def update(
        self,
        agents: list,
        agent_names: list[str],
        routed_messages: dict[str, list[Message]],
        agent_outputs: Optional[dict[str, AgentOutput]] = None,
        round_num: int = 0,
        aggregation_order: Optional[list[int]] = None,
        context=None,
    ) -> None:
        _ = context  # message_log 策略不依赖上下文
        # 自动识别 Manager（role 以 Workflow Orchestrator 开头）
        manager_name = None
        for agent in agents:
            role = getattr(agent, "role", "")
            if role and role.startswith("Workflow Orchestrator"):
                manager_name = agent.name
                break

        # 从 agent_outputs 构建本轮全局契约账本
        ledger_entries = []
        for name, output in (agent_outputs or {}).items():
            if not output or not output.private_routing:
                continue
            for item in output.private_routing:
                if not isinstance(item, dict):
                    continue
                rcpts = item.get("recipients", [])
                if isinstance(rcpts, str):
                    rcpts = [rcpts]
                for rcpt in rcpts:
                    ledger_entries.append({
                        "round": round_num,
                        "sender": name,
                        "receiver": rcpt,
                        "illocution": item.get("illocution", "Assertive"),
                        "content": str(item.get("content", ""))[:200],
                    })

        for agent in agents:
            private_msgs = []
            for msg in routed_messages.get(agent.name, []):
                if msg.channel in self.SUPPORTED_CHANNELS:
                    private_msgs.append(msg)

            if aggregation_order is not None and private_msgs:
                name_to_idx = {name: idx for idx, name in enumerate(agent_names)}

                def _order_key(msg: Message):
                    sender_idx = name_to_idx.get(msg.sender_id, len(agent_names))
                    try:
                        pos = aggregation_order.index(sender_idx)
                    except ValueError:
                        pos = len(aggregation_order)
                    return pos

                private_msgs.sort(key=_order_key)

            # 写入自身输出摘要（private_routing）
            if agent_outputs and agent.name in agent_outputs:
                output = agent_outputs[agent.name]
                own_parts = []
                if output.private_routing:
                    for item in output.private_routing:
                        if not isinstance(item, dict):
                            continue
                        rcpts = item.get("recipients", [])
                        if isinstance(rcpts, str):
                            rcpts = [rcpts]
                        illoc = item.get("illocution", "Assertive")
                        content = item.get("content", "")
                        own_parts.append(
                            f"[{illoc} -> {', '.join(rcpts)}]: {content}"
                        )
                if own_parts:
                    agent.append_memory(
                        f"[ROUND {round_num} - My Output]:\n" + "\n".join(own_parts)
                    )
                elif output.answer and str(output.answer).strip():
                    agent.append_memory(
                        f"[ROUND {round_num} - My Output]:\n[Answer]: {output.answer}"
                    )

            # 写入收到的 private / broadcast 消息
            for msg in private_msgs:
                channel_label = "broadcast" if msg.channel == "ft-broadcast" else msg.channel
                agent.append_memory(
                    f"[From {msg.sender_id} ({channel_label})]: {msg.content}"
                )

            # Manager 额外注入全局契约账本摘要
            if agent.name == manager_name and ledger_entries:
                ledger_lines = [f"--- Global Contract Ledger (Round {round_num}) ---"]
                for entry in ledger_entries:
                    if entry.get("round") == round_num:
                        ledger_lines.append(
                            f"[Log] {entry['sender']} -> {entry['receiver']} "
                            f"[{entry['illocution']}]: {entry['content']}"
                        )
                if len(ledger_lines) > 1:
                    agent.append_memory("\n".join(ledger_lines))
