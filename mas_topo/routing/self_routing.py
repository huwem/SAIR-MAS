"""Self Routing Strategy — PragmaMAS-P2P 纯语旨自路由策略。

核心逻辑：
1. 所有 Agent 通过 private_routing 数组自主声明消息的接收者、语旨类别和内容。
2. 框架仅做解析、邻居约束校验（基于当前轮次的邻接矩阵）和投递。
3. 每条语旨行为都被写入共享的 IllocutionaryLedger，供后续状态机审计。
4. 选择性激活：基于上一轮 private_routing 的 recipients 决定本轮激活集合。
5. 多跳通信由 Agent 通过标准语旨（Directive/Assertive/Commissive/Expressive/Declarative）
   社会性地组织；框架只负责单跳投递与 ledger 记录，不引入非标准语旨类别。
6. 不再依赖 Manager 终止信号；终止由 ConsensusDecision 根据 ledger 健康与投票决定。
"""

import logging
from typing import Optional

import numpy as np

from mas_topo._registries import routing_registry
from mas_topo.context import ExecutionContext
from mas_topo.ledger import IllocutionaryLedger
from mas_topo.message import AgentOutput, Channel, Message
from mas_topo.routing.base import RoutingStrategy
from mas_topo.tools.builtin import RUN_PYTHON_TOOL_NAME

logger = logging.getLogger(__name__)


class SelfRouting(RoutingStrategy):
    """自路由策略。

    Agent 通过 private_routing 自主声明通信对象和语旨，框架仅做解析、
    邻居校验、投递，并将行为写入共享 ledger。

    多跳场景下，框架不自动转发；Agent 需要利用注入记忆中的局部邻居表与
    网络认知视图，通过标准语旨（如 Directive）请某位直接邻居代为转发。

    Args:
        memory_truncate: 消息内容截断长度（0 表示不截断）。
        manager_name: 兼容旧配置，P2P 中通常不设置。
        neighbor_policy: 邻居校验策略，"strict" | "filter" | "permissive"。
        agent_neighbor_map: 兼容旧配置；若提供，在无法获得邻接矩阵时作为邻居来源。
    """

    supports_selective_activation = True

    def __init__(
        self,
        memory_truncate: int = 3000,
        manager_name: Optional[str] = None,
        neighbor_policy: str = "strict",
        agent_neighbor_map: Optional[dict[str, list[str]]] = None,
        max_out_degree: Optional[int] = None,
    ):
        self.memory_truncate = memory_truncate
        self.manager_name = manager_name
        self.neighbor_policy = neighbor_policy
        self.agent_neighbor_map = agent_neighbor_map or {}
        # 每个 Agent 每轮所有 private_routing 项加在一起，最多能发给多少个不同的接收者。
        # None 表示不限制；设为 2 时，广播（All）会被截断/拒绝。
        self.max_out_degree = max_out_degree

    def _allowed_neighbors_from_adj(
        self, sender_name: str, agent_names: list[str], adjacency_matrix: np.ndarray
    ) -> set[str]:
        """从邻接矩阵获取发送者被允许的邻居集合。"""
        if adjacency_matrix is None or adjacency_matrix.size == 0:
            return set()
        n = len(agent_names)
        if adjacency_matrix.shape != (n, n):
            return set()
        sender_idx = agent_names.index(sender_name)
        return {
            agent_names[j]
            for j in range(n)
            if adjacency_matrix[sender_idx][j] == 1
        }

    def _allowed_neighbors_from_map(self, sender_name: str) -> set[str]:
        """从显式邻居映射获取允许邻居集合。"""
        return set(self.agent_neighbor_map.get(sender_name, []))

    def _resolve_recipient(
        self,
        recipient: str,
        sender_name: str,
        allowed: set[str],
        agent_names: list[str],
    ) -> Optional[str]:
        """解析并校验单个接收者名称。

        支持容错：
        - "all"/"ALL" 等大小写变体统一为 "All"
        - "_self" 或发送者自身名称映射为 sender_name
        - 接收者名称大小写不敏感，自动匹配到 agent_names 中的正确大小写
        - 未知或越界名称返回 None（单条丢弃，不影响同组其他接收者）

        Returns:
            标准化后的接收者名称，或 None 表示该接收者无效。
        """
        if not isinstance(recipient, str) or not recipient.strip():
            return None

        lower_map = {name.lower(): name for name in agent_names}
        raw = recipient.strip()

        # 容错：去掉 LLM 可能附带的角色说明，如 "Tester (Code Tester)"
        # 也兼容 "Name - Role"、"Name: Role" 等常见变体
        cleaned = raw
        for sep in ("(", " -", ":", "<", "["):
            if sep in cleaned:
                cleaned = cleaned.split(sep, 1)[0]
        cleaned = cleaned.strip()
        lower = cleaned.lower()

        if lower == "all":
            return "All"
        if lower == "_self":
            return sender_name

        # 大小写不敏感匹配到已知 Agent
        canonical = lower_map.get(lower)
        if canonical is None:
            logger.warning(
                "SelfRouting: %s tried to send to unknown recipient '%s' (ignored)",
                sender_name, raw,
            )
            return None

        # 自己给自己发始终允许
        if canonical == sender_name:
            return canonical

        if self.neighbor_policy == "permissive":
            return canonical

        if canonical not in allowed:
            logger.warning(
                "SelfRouting: %s tried to send to non-neighbor '%s' (ignored)",
                sender_name, canonical,
            )
            return None

        return canonical

    def _filter_recipients(
        self,
        sender_name: str,
        recipients: list[str],
        allowed: set[str],
        agent_names: list[str],
    ) -> list[str]:
        """根据 neighbor_policy 过滤接收者。

        与旧版 strict 模式不同：现在只丢弃无效/越界接收者，保留同组中合法的接收者，
        并支持自己给自己发信息。

        Returns:
            过滤并标准化后的接收者列表（不含 "All" 通配符）。
        """
        resolved: list[str] = []
        seen = set()
        for r in recipients:
            name = self._resolve_recipient(r, sender_name, allowed, agent_names)
            if name is not None and name != "All" and name not in seen:
                resolved.append(name)
                seen.add(name)
        return resolved

    def _parse_private_routing(self, output: AgentOutput) -> list[dict]:
        """解析 AgentOutput 中的 private_routing 字段。

        Returns:
            标准化后的路由指令列表，每项为 {"recipients": [str], "illocution": str, "content": str}。
        """
        routing = output.private_routing or []
        normalized = []
        for item in routing:
            if not isinstance(item, dict):
                continue
            recipients = item.get("recipients", [])
            if isinstance(recipients, str):
                recipients = [recipients]
            elif not isinstance(recipients, list):
                continue
            normalized.append({
                "recipients": list(set(recipients)),
                "illocution": str(item.get("illocution", "Assertive")).strip(),
                "content": str(item.get("content", "")).strip(),
                # 兼容新旧字段名：answer_correct（新）/ vote_complete（旧）
                "vote_complete": bool(item.get("answer_correct", item.get("vote_complete", False))),
            })
        return normalized

    def route(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
        aggregation_order=None,
    ) -> dict[str, list[Message]]:
        """执行自路由，返回各 Agent 收到的消息字典。"""
        routed_messages, _ = self.route_and_log(
            agent_names, agent_outputs, adjacency_matrix, context, aggregation_order
        )
        return routed_messages

    def _get_allowed_neighbors(
        self,
        src_name: str,
        agent_names: list[str],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
    ) -> set[str]:
        """获取发送者被允许的邻居集合。

        优先使用 Agent 统一局部邻居表中的自身条目（local_neighbor_table[src_name]），
        即使为空也不回退，以保证外部拓扑变化的言后效果不会被全局邻接矩阵覆盖；
        只有当 Agent 没有 local_neighbor_table 属性时，才回退到全局邻接矩阵或显式 map。
        """
        agents = getattr(context, "agents", []) or []
        for agent in agents:
            if agent.name == src_name:
                if hasattr(agent, "local_neighbor_table"):
                    return set(agent.local_neighbor_table.get(src_name, set()))
                break

        allowed = self._allowed_neighbors_from_adj(src_name, agent_names, adjacency_matrix)
        if not allowed and self.agent_neighbor_map:
            allowed = self._allowed_neighbors_from_map(src_name)
        return allowed

    def route_and_log(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: np.ndarray,
        context: ExecutionContext,
        aggregation_order=None,
    ) -> tuple[dict[str, list[Message]], list]:
        """执行自路由并写入共享 ledger。

        局部邻居表由外部拓扑策略维护并注入到每个 Agent；本函数只读取
        Agent.local_neighbors 执行单跳邻居校验，不根据 Agent 语旨增删邻居。

        Returns:
            (routed_messages, ledger_entries)
            - routed_messages: {agent_name: [Message, ...]}
            - ledger_entries: [LedgerEntry, ...]
        """
        result: dict[str, list[Message]] = {name: [] for name in agent_names}
        ledger_entries: list = []
        trunc = self.memory_truncate
        round_num = getattr(context, "round_num", 0)

        # 使用策略选定的状态后端；不存在则创建并挂载到 context.ledger
        state = self._state_backend(context)
        if state is None:
            state = IllocutionaryLedger()
            context.ledger = state

        for src_name in agent_names:
            src_out = agent_outputs.get(src_name)
            if not src_out:
                continue

            allowed = self._get_allowed_neighbors(
                src_name, agent_names, adjacency_matrix, context
            )

            routing_items = self._parse_private_routing(src_out)
            if not routing_items:
                continue

            # 追踪该发送者本轮已经使用的接收者，用于限制总出度。
            sender_used_recipients: set[str] = set()

            for item in routing_items:
                illocution = item["illocution"].strip()

                raw_recipients = item["recipients"]
                is_broadcast = any(
                    str(r).strip().lower() == "all" for r in raw_recipients
                )

                normal_recipients = self._filter_recipients(
                    src_name, raw_recipients, allowed, agent_names
                )

                if is_broadcast:
                    broadcast_targets = [n for n in agent_names if n != src_name]
                    all_recipients = list(dict.fromkeys(normal_recipients + broadcast_targets))
                else:
                    all_recipients = list(dict.fromkeys(normal_recipients))

                if not all_recipients:
                    continue

                # ── 出度硬限制 ──────────────────────────────────────────
                # 每个 Agent 每轮所有 private_routing 项加在一起，最多发给 max_out_degree 个不同接收者。
                # 超出部分按出现顺序截断，并写 warning log。
                if self.max_out_degree is not None:
                    remaining = max(0, self.max_out_degree - len(sender_used_recipients))
                    if remaining == 0:
                        logger.warning(
                            "SelfRouting: %s has already reached max_out_degree=%d; "
                            "dropping routing item to %s",
                            src_name, self.max_out_degree, all_recipients,
                        )
                        continue
                    if len(all_recipients) > remaining:
                        dropped = all_recipients[remaining:]
                        all_recipients = all_recipients[:remaining]
                        logger.warning(
                            "SelfRouting: %s routing item exceeds remaining out-degree budget %d; "
                            "keeping %s, dropping %s",
                            src_name, remaining, all_recipients, dropped,
                        )
                    sender_used_recipients.update(all_recipients)

                content = item["content"]
                formatted_content = f"[{illocution}]: {content}"
                if trunc and trunc > 0:
                    formatted_content = formatted_content[:trunc]

                # 发送者当前 answer 快照：不再拼到消息正文，而是写入 ledger 供动态注入
                src_answer = str(src_out.answer) if src_out and src_out.answer else ""

                channel = Channel.BROADCAST if is_broadcast else Channel.FT_PRIVATE

                for dst_name in all_recipients:
                    if dst_name not in agent_names:
                        continue
                    result[dst_name].append(
                        Message(
                            content=formatted_content,
                            sender_id=src_name,
                            receiver_id=dst_name,
                            channel=channel,
                            metadata={
                                "illocution_category": illocution,
                                "illocution_content": content,
                                "is_broadcast": is_broadcast,
                            },
                        )
                    )

                # 写入共享 ledger：每个 routing 项一条记录，附带 answer 快照
                # 若该发送者本轮有测试工具调用，取最新结果作为 verification_status
                verification_status = self._extract_answer_verification_status(
                    src_out
                )
                entry = state.append(
                    round_num=round_num,
                    sender=src_name,
                    recipients=all_recipients,
                    illocution=illocution,
                    content=content,
                    answer_snapshot=src_answer if src_answer.strip() else None,
                    verification_status=verification_status,
                )
                ledger_entries.append(entry)

        return result, ledger_entries

    @staticmethod
    def _extract_answer_verification_status(src_out: AgentOutput) -> Optional[str]:
        """从 Agent 本轮的工具调用中提取其 answer 对应的验证状态。

        仅关注 run_python 工具的最新结果。返回 "PASSED"/"FAILED"/None。
        这里使用工具结果中的 passed 字段，而不是 stderr 内容，避免误判。
        """
        tool_calls = getattr(src_out, "tool_calls", None) or []
        latest_status: Optional[str] = None
        for call_info in tool_calls:
            if not isinstance(call_info, dict):
                continue
            call = call_info.get("call", {}) or {}
            result = call_info.get("result", {}) or {}
            if call.get("name") != RUN_PYTHON_TOOL_NAME:
                continue
            result_data = result.get("result") if isinstance(result, dict) else result
            if isinstance(result_data, dict):
                passed = result_data.get("passed")
                if passed is True:
                    latest_status = "PASSED"
                elif passed is False:
                    latest_status = "FAILED"
                elif passed is None and result_data.get("error"):
                    latest_status = "ERROR"
        return latest_status

    def check_termination(
        self,
        agent_outputs: dict[str, AgentOutput],
    ) -> bool:
        """（兼容旧代码）检查 Manager 是否发出 TASK_COMPLETE 声明。

        P2P 模式下终止由 ConsensusDecision 负责，此接口保留以避免破坏旧调用。
        """
        if not self.manager_name:
            return False

        manager_out = agent_outputs.get(self.manager_name)
        if not manager_out:
            return False

        routing_items = self._parse_private_routing(manager_out)
        for item in routing_items:
            recipients = item.get("recipients", [])
            if isinstance(recipients, str):
                recipients = [recipients]
            if (
                "All" in recipients
                and item.get("illocution", "").lower() == "declarative"
                and item.get("content", "").strip().upper() == "TASK_COMPLETE"
            ):
                logger.info(
                    "SelfRouting: termination signal detected from %s: "
                    "[Declarative] to All: TASK_COMPLETE",
                    self.manager_name,
                )
                return True

        return False

    def get_active_workers(
        self,
        context: ExecutionContext,
        worker_names: list[str],
        previous_outputs: Optional[dict[str, AgentOutput]] = None,
        **kwargs,
    ) -> set[str]:
        """基于上一轮 private_routing 的 recipients 决定激活 Worker。

        首轮（previous_outputs 为空）激活答案产出角色（Developer / Solver），
        既避免唤醒所有 Agent，也不把流程串行化为只有单一节点工作。
        后续轮次只激活被指定为接收者的 Agent（支持 self-messaging）。
        若某轮没有合法接收者，则回退到激活全部 Worker，确保不会死锁。
        """
        if not previous_outputs:
            # 首轮种子激活：答案产出角色（Developer / Solver）
            # 不再依赖固定 Coordinator，体现纯 P2P 设计
            seed = set()
            for role in ("Developer", "Solver"):
                if role in worker_names:
                    seed.add(role)
            if seed:
                logger.info("SelfRouting: first round, activating seed workers %s", sorted(seed))
                return seed
            logger.info("SelfRouting: first round, no answer role found; activating all")
            return set(worker_names)

        active = set()
        directive_forced = set()

        for agent_name, output in previous_outputs.items():
            if not output:
                continue
            routing_items = self._parse_private_routing(output)
            for item in routing_items:
                for raw_recipient in item["recipients"]:
                    # 允许所有 Agent 作为潜在接收者（激活阶段不限于邻居）
                    resolved = self._resolve_recipient(
                        raw_recipient, agent_name, set(worker_names), worker_names
                    )
                    if resolved is None:
                        continue
                    if resolved == "All":
                        active.update(worker_names)
                    else:
                        active.add(resolved)
                    if item["illocution"].lower() == "directive":
                        if resolved == "All":
                            directive_forced.update(worker_names)
                        else:
                            directive_forced.add(resolved)

        if not active:
            logger.info("SelfRouting: no active recipients, fallback to all workers")
            return set(worker_names)

        if directive_forced:
            logger.info(
                "SelfRouting: Directive forced activation for %s",
                sorted(directive_forced),
            )

        return active


@routing_registry.register(
    "self_routing",
    display_name="自路由",
    description="Agent 通过 private_routing 自主声明通信对象和语旨，框架做邻居校验、投递并写入共享 ledger。多跳由 Agent 通过标准语旨社会性组织。",
    config_schema={
        "memory_truncate": {
            "type": "int",
            "default": 3000,
            "description": "消息内容截断长度，0 表示不截断",
        },
        "manager_name": {
            "type": "str|null",
            "default": None,
            "description": "兼容旧配置，P2P 中通常不设置",
        },
        "neighbor_policy": {
            "type": "str",
            "default": "strict",
            "description": "邻居校验策略: strict | filter | permissive",
        },
        "agent_neighbor_map": {
            "type": "object|null",
            "default": None,
            "description": "兼容旧配置：显式邻居映射 {agent_name: [neighbor_name, ...]}",
        },
        "max_out_degree": {
            "type": "int|null",
            "default": None,
            "description": "每个 Agent 每轮所有 private_routing 项最多能发给多少个不同接收者；None 表示不限制",
        },
    },
)
class SelfRoutingRegistered(SelfRouting):
    """注册到全局注册表的版本。"""

    def __init__(
        self,
        memory_truncate: int = 3000,
        manager_name: Optional[str] = None,
        neighbor_policy: str = "strict",
        agent_neighbor_map: Optional[dict[str, list[str]]] = None,
        max_out_degree: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            memory_truncate=memory_truncate,
            manager_name=manager_name,
            neighbor_policy=neighbor_policy,
            agent_neighbor_map=agent_neighbor_map,
            max_out_degree=max_out_degree,
        )
