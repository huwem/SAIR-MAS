"""AutoGen GroupChat 风格路由策略（对照官方 v0.2 实现）。

每轮由 Manager（Coordinator）选择一个 Worker 发言，该 Worker 的回复广播给
所有其他 Agent（包括 Manager），从而模拟 AutoGen 论文中 GroupChatManager 的
"select a single speaker -> collect response -> broadcast" 行为。

- 首轮发言人：与原版一致由 LLM 从仅含任务的 transcript 中选择
  （需注入 llm_adapter；缺失时退化为 role_order 兜底）。
- 后续发言人：由 Manager 上一轮决策输出的 `next_speaker` 字段决定；
  无效时按原版 next_agent 语义取 last_speaker 的下一个。
"""

from typing import Optional

import numpy as np

from mas_topo._registries import routing_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import Message, AgentOutput
from mas_topo.routing.base import RoutingStrategy
from mas_topo.decision.autogen_manager import select_speaker_auto


class AutoGenGroupChatRouting(RoutingStrategy):
    """AutoGen GroupChat 风格路由：单发言人 + 广播。

    - 支持选择性激活（supports_selective_activation=True），每轮只激活一个 Worker。
    - 激活 Worker 由 Manager 上一轮的 `next_speaker` 字段决定；无效时按
      `role_order` 轮询兜底。
    - 被激活 Worker 的 public_content / answer 会广播给其余所有 Agent。
    """

    supports_selective_activation = True
    # 原版 AutoGen GroupChatManager 不在 transcript 中产出内容，
    # 跳过 Round 0 Manager 初始推理；第 1 轮发言人由 role_order 兜底。
    requires_manager_initial = False

    def __init__(
        self,
        role_order: Optional[list[str]] = None,
        fallback_mode: str = "round_robin",
        include_manager: bool = True,
        llm_adapter=None,
    ):
        self.role_order = list(role_order) if role_order else []
        self.fallback_mode = fallback_mode
        self.include_manager = include_manager
        # 原版 AutoGen 首个发言人也由 LLM 选择（transcript 仅含任务）；
        # llm_adapter 由框架注入（仅 autogen_groupchat），缺失时退化为
        # role_order 兜底。
        self.llm_adapter = llm_adapter

    def _manager_next_speaker(
        self,
        previous_outputs: Optional[dict[str, AgentOutput]],
        worker_names: list[str],
    ) -> Optional[str]:
        """从上一轮的 Manager 输出中解析 next_speaker。"""
        if not previous_outputs:
            return None
        for name, out in previous_outputs.items():
            if not out:
                continue
            speaker = getattr(out, "next_speaker", None)
            if not speaker:
                # 兼容：也尝试从 next_goal 中读取合法 worker 名
                speaker = getattr(out, "next_goal", None)
            if speaker and speaker in worker_names:
                return speaker
        return None

    def _last_speaker(
        self,
        previous_outputs: Optional[dict[str, AgentOutput]],
        worker_names: list[str],
    ) -> Optional[str]:
        """上一轮发言人：唯一产生内容的 Worker。"""
        if not previous_outputs:
            return None
        for name in worker_names:
            out = previous_outputs.get(name)
            if out and (out.public_content or (out.answer and str(out.answer).strip())):
                return name
        return None

    def _next_after(
        self, last_speaker: Optional[str], order: list[str]
    ) -> Optional[str]:
        """原版 GroupChat.next_agent：last_speaker 在名册中的下一个。"""
        if not order:
            return None
        if last_speaker in order:
            idx = order.index(last_speaker)
            return order[(idx + 1) % len(order)]
        return order[0]

    def get_active_workers(
        self,
        context: ExecutionContext,
        worker_names: list[str],
        previous_outputs: Optional[dict[str, AgentOutput]] = None,
        **kwargs,
    ) -> set[str]:
        """每轮只激活一个 Worker（GroupChat 发言人）。"""
        if not worker_names:
            return set()

        # 优先采纳 Manager 指定的 next_speaker
        manager_speaker = self._manager_next_speaker(previous_outputs, worker_names)
        if manager_speaker:
            return {manager_speaker}

        last_speaker = self._last_speaker(previous_outputs, worker_names)

        # 首轮（无任何历史输出）：原版由 LLM 从仅含任务的 transcript 中选择
        if previous_outputs is None and self.llm_adapter is not None:
            selected = select_speaker_auto(
                self.llm_adapter,
                [{"role": "user", "content": context.task}] if context.task else [],
                context.team_roster,
                worker_names,
                context=context,
            )
            if selected:
                return {selected}

        # 兜底：按 role_order 取 last_speaker 的下一个（原版 next_agent 语义）
        order = [r for r in self.role_order if r in worker_names]
        if order:
            nxt = self._next_after(last_speaker, order)
            if nxt:
                return {nxt}

        if self.fallback_mode == "all":
            return set(worker_names)

        # 默认：全体 worker 名册中 last_speaker 的下一个
        nxt = self._next_after(last_speaker, worker_names)
        return {nxt} if nxt else set()

    def route(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
        aggregation_order=None,
    ) -> dict[str, list[Message]]:
        """将本轮发言人的 public_content/answer 广播给其余所有 Agent。"""
        result = {name: [] for name in agent_names}

        # 本轮实际产生了输出的 Worker 即发言人（Graph 已经通过选择性激活只运行了一个 Worker）
        speaker = None
        src_out = None
        for name in agent_names:
            out = agent_outputs.get(name)
            if out and (out.public_content or out.answer):
                speaker = name
                src_out = out
                break
        if not speaker or not src_out:
            return result

        contents: list[tuple[str, str]] = []
        if src_out.public_content:
            contents.append((src_out.public_content, "broadcast"))
        if src_out.answer:
            contents.append((src_out.answer, "answer"))

        for dst_name in agent_names:
            if dst_name == speaker:
                continue
            # 在 GroupChat 中消息对所有人可见，不遵守物理邻接矩阵
            for content, channel in contents:
                result[dst_name].append(Message(
                    content=content,
                    sender_id=speaker,
                    receiver_id=dst_name,
                    channel=channel,
                ))

        return result


@routing_registry.register(
    "autogen_groupchat",
    display_name="AutoGen GroupChat 路由",
    description="每轮由 Manager 选择一个 Worker 发言并广播其回复，模拟 AutoGen GroupChatManager。",
    config_schema={
        "role_order": {
            "type": "list[str]",
            "default": [],
            "description": "Worker 默认发言顺序（兜底轮询用）",
        },
        "fallback_mode": {
            "type": "str",
            "default": "round_robin",
            "description": "Manager 未指定 next_speaker 时的兜底策略：round_robin / all",
        },
        "include_manager": {
            "type": "bool",
            "default": True,
            "description": "是否把消息也广播给 Manager",
        },
    },
)
class AutoGenGroupChatRoutingRegistered(AutoGenGroupChatRouting):
    """注册到全局注册表的版本。"""

    def __init__(
        self,
        role_order: Optional[list] = None,
        fallback_mode: str = "round_robin",
        include_manager: bool = True,
        llm_adapter=None,
        **kwargs,
    ):
        super().__init__(
            role_order=role_order or [],
            fallback_mode=fallback_mode,
            include_manager=include_manager,
            llm_adapter=llm_adapter,
        )
