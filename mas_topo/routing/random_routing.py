"""随机路由策略。

用于基线对比：每轮随机选择激活的 Worker，并随机选择消息接收者，
以验证语旨路由的有效性和效率优势。
"""

import logging
import random

import numpy as np

from mas_topo._registries import routing_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import Message, AgentOutput
from mas_topo.routing.base import RoutingStrategy

logger = logging.getLogger(__name__)


class RandomRouting(RoutingStrategy):
    """随机路由策略。

    每轮按一定概率随机激活 Worker；对于每条 outgoing 消息，
    在物理邻接约束下随机选择接收者。
    """

    supports_selective_activation = True

    def __init__(
        self,
        activation_prob: float = 0.7,
        min_active: int = 2,
        recipient_prob: float = 0.5,
        route_public: bool = True,
        random_seed: int = 42,
        first_round_full: bool = False,
    ):
        self.activation_prob = activation_prob
        self.min_active = min_active
        self.recipient_prob = recipient_prob
        self.route_public = route_public
        self.random_seed = random_seed
        self.first_round_full = first_round_full

    def _rng(self, context: ExecutionContext, phase: int = 0):
        """基于轮次和阶段生成确定性随机数生成器。

        phase 用于隔离不同阶段的随机流：
        0 = 激活选择，1 = 路由选择，避免同轮内共享随机序列导致相关性。
        """
        seed = self.random_seed + context.round_num * 1000 + phase
        return random.Random(seed)

    def get_active_workers(
        self,
        context: ExecutionContext,
        worker_names: list[str],
        previous_outputs: dict | None = None,
        **kwargs,
    ) -> set[str]:
        """随机选择激活的 Worker。"""
        rng = self._rng(context, phase=0)
        # 默认按概率随机激活每一轮（包括首轮）。
        # 若设置 first_round_full=true，则首轮或 previous_outputs 为空时激活全部，避免初始死锁。
        if self.first_round_full and not previous_outputs:
            return set(worker_names)

        active = set()
        for name in worker_names:
            if rng.random() < self.activation_prob:
                active.add(name)
        if len(active) < self.min_active and worker_names:
            # 随机补充到 min_active
            remaining = [n for n in worker_names if n not in active]
            rng.shuffle(remaining)
            for name in remaining:
                active.add(name)
                if len(active) >= self.min_active:
                    break

        inactive = set(worker_names) - active
        logger.info(
            "RandomRouting round %d: active=%s inactive=%s (activation_prob=%.2f, min_active=%d)",
            context.round_num,
            sorted(active),
            sorted(inactive),
            self.activation_prob,
            self.min_active,
        )
        return active

    def route(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
        aggregation_order=None,
    ) -> dict[str, list[Message]]:
        """随机选择接收者并投递消息。"""
        rng = self._rng(context, phase=1)
        N = len(agent_names)
        result = {name: [] for name in agent_names}

        name_to_idx = {name: i for i, name in enumerate(agent_names)}

        for src_name in agent_names:
            src_out = agent_outputs.get(src_name)
            if not src_out:
                continue

            # 收集 outgoing 消息内容
            outgoing: list[tuple[str, str, str]] = []  # (content, channel, inferred_illocution)
            for item in getattr(src_out, "private_routing", []) or []:
                if not isinstance(item, dict):
                    continue
                content = item.get("content", "")
                if content:
                    outgoing.append((content, "ft-private", item.get("illocution", "")))

            if self.route_public and src_out.public_content:
                outgoing.append((src_out.public_content, "broadcast", ""))

            if not outgoing:
                continue

            src_idx = name_to_idx[src_name]
            # 候选邻居（排除自己，受邻接矩阵约束）
            candidates = []
            for dst_idx in range(N):
                if dst_idx == src_idx:
                    continue
                if adjacency_matrix is not None and adjacency_matrix[src_idx][dst_idx] == 0:
                    continue
                candidates.append(agent_names[dst_idx])

            if not candidates:
                continue

            for content, channel, illoc in outgoing:
                # 对候选接收者按 recipient_prob 随机选择
                chosen = [dst for dst in candidates if rng.random() < self.recipient_prob]
                if not chosen and candidates:
                    # 至少保证一个接收者
                    chosen = [rng.choice(candidates)]
                for dst_name in chosen:
                    result[dst_name].append(Message(
                        content=content,
                        sender_id=src_name,
                        receiver_id=dst_name,
                        channel=channel,
                        metadata={"illocution": illoc},
                    ))

        return result


@routing_registry.register(
    "random",
    display_name="随机路由",
    description="随机激活 Worker 并随机选择消息接收者，用于基线对比。",
    config_schema={
        "activation_prob": {
            "type": "float",
            "default": 0.7,
            "description": "每个 Worker 被激活的概率",
        },
        "min_active": {
            "type": "int",
            "default": 2,
            "description": "每轮最少激活的 Worker 数",
        },
        "recipient_prob": {
            "type": "float",
            "default": 0.5,
            "description": "每个邻居被选为接收者的概率",
        },
        "route_public": {
            "type": "bool",
            "default": True,
            "description": "是否将 public_content 也随机路由",
        },
        "random_seed": {
            "type": "int",
            "default": 42,
            "description": "随机种子",
        },
        "first_round_full": {
            "type": "bool",
            "default": False,
            "description": "首轮是否激活全部 Worker（true 为兼容旧行为；false 则首轮也按概率随机激活）",
        },
    },
)
class RandomRoutingRegistered(RandomRouting):
    """注册到全局注册表的随机路由版本。"""

    def __init__(
        self,
        activation_prob: float = 0.7,
        min_active: int = 2,
        recipient_prob: float = 0.5,
        route_public: bool = True,
        random_seed: int = 42,
        first_round_full: bool = False,
        **kwargs,
    ):
        super().__init__(
            activation_prob=activation_prob,
            min_active=min_active,
            recipient_prob=recipient_prob,
            route_public=route_public,
            random_seed=random_seed,
            first_round_full=first_round_full,
        )
