"""统一多智能体执行引擎。

实现 5 阶段执行循环：
1. Agent Inference — 所有 Agent 推理
2. Topology Construction — 拓扑策略构建邻接矩阵
3. Message Routing — 路由策略计算消息传递
4. Memory Update — 更新 Agent 记忆
5. Decision — 决策策略判断是否完成
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from tenacity import RetryError

from mas_topo.context import ExecutionContext
from mas_topo.ledger import IllocutionaryLedger
from mas_topo.message import AgentOutput, Message
from mas_topo.decision.base import DecisionResult, DecisionStrategy
from mas_topo.routing.base import RoutingStrategy
from mas_topo.topology.base import TopologyStrategy
from mas_topo.memory.base import MemoryUpdateStrategy
from mas_topo.memory.default import DefaultMemoryUpdate

logger = logging.getLogger(__name__)


@dataclass
class RoundResult:
    """单轮执行结果。"""
    round_num: int = 0
    agent_outputs: dict[str, AgentOutput] = field(default_factory=dict)
    adjacency_matrix: Optional[np.ndarray] = None
    relevance_matrix: Optional[np.ndarray] = None
    routed_messages: dict[str, list[Message]] = field(default_factory=dict)
    decision: Optional[DecisionResult] = None
    elapsed_time: float = 0.0
    ledger: Optional[Any] = field(default=None)


@dataclass
class GraphResult:
    """Graph 执行结果。"""
    final_answer: Optional[str] = None
    rounds: list[RoundResult] = field(default_factory=list)
    context: ExecutionContext = field(default_factory=ExecutionContext)
    is_completed: bool = False


class Graph:
    """统一多智能体执行引擎。

    参数:
        agents: Agent 列表（可为空列表，此时需提供 task_designer）。
        topology_strategy: 拓扑构建策略。
        decision_strategy: 决策策略。
        routing_strategy: 消息路由策略。
        max_rounds: 最大执行轮次。
        max_retries: Agent 执行失败最大重试次数。
        verbose: 是否输出详细日志。
        topology_logger: 可选的拓扑日志记录器。
        task_designer: 可选的 TaskDesigner，在 pre-execution 阶段根据 task 动态设计
            Agent 组合和初始拓扑。传入后 agents 可为空列表，Agent 由 designer 动态生成。
    """

    def __init__(
        self,
        agents: list,
        topology_strategy: TopologyStrategy,
        decision_strategy: DecisionStrategy,
        routing_strategy: RoutingStrategy,
        max_rounds: int = 10,
        max_retries: int = 3,
        verbose: bool = True,
        topology_logger: Optional[Any] = None,
        task_designer: Optional[Any] = None,
        initial_goal: Optional[str] = None,
        memory_strategy: Optional[MemoryUpdateStrategy] = None,
    ):
        self.agents = list(agents)
        self.topology_strategy = topology_strategy
        self.decision_strategy = decision_strategy
        self.routing_strategy = routing_strategy
        self.max_rounds = max_rounds
        self.max_retries = max_retries
        self.verbose = verbose
        self.topology_logger = topology_logger
        self.task_designer = task_designer
        self.initial_goal = initial_goal
        self.memory_strategy = memory_strategy or DefaultMemoryUpdate()

        self.agent_names = [a.name for a in agents]
        self.name_to_agent = {a.name: a for a in agents}

    # ── 同步执行 ──────────────────────────────────────────

    def run(self, task: str, **kwargs) -> GraphResult:
        """同步执行。

        委托给 arun()，所有 Agent 在单个 asyncio 事件循环中通过
        asyncio.gather 并行执行，无需 ThreadPoolExecutor。
        """
        import asyncio
        return asyncio.run(self.arun(task, **kwargs))

    def _apply_task_design(self, design) -> None:
        """应用 TaskDesigner 的输出：动态创建 Agent 并设置初始拓扑。"""
        import numpy as np
        from mas_topo.topology.manual import ManualTopology
        from mas_topo._registries import agent_registry

        new_agents = []
        for cfg in design.agent_configs:
            llm = getattr(cfg, 'llm_adapter', None)
            if llm is None and self.agents:
                llm = getattr(self.agents[0], 'llm_adapter', None)
            params = dict(cfg.params) if hasattr(cfg, 'params') and cfg.params else {}
            if llm is not None:
                params.setdefault('llm_adapter', llm)
            agent = agent_registry.get(
                cfg.type,
                name=cfg.name or '',
                role=cfg.role or '',
                system_prompt=getattr(cfg, 'system_prompt', None),
                output_format=getattr(cfg, 'output_format', 'text'),
                output_schema=getattr(cfg, 'output_schema', None),
                tools=getattr(cfg, 'tools', None) or [],
                **params,
            )
            new_agents.append(agent)

        self.agents = new_agents
        self.agent_names = [a.name for a in new_agents]
        self.name_to_agent = {a.name: a for a in new_agents}

        if design.adjacency_matrix is not None:
            adj = np.array(design.adjacency_matrix)
            if adj.shape == (len(new_agents), len(new_agents)):
                self.topology_strategy = ManualTopology(adjacency=adj)

    # ── 异步执行 ──────────────────────────────────────────

    async def arun(self, task: str, **kwargs) -> GraphResult:
        """异步执行。

        Keyword Args:
            initial_goal: 初始轮次目标。
            metadata: 附加到 ExecutionContext 的字典。
        """
        # ── Pre-execution: Task-conditioned system design ─────────
        if self.task_designer is not None:
            design = self.task_designer.design(task)
            if design.agent_configs:
                self._apply_task_design(design)
                logger.info(
                    "TaskDesigner: designed %d agents, %d edges for task",
                    len(design.agent_configs),
                    int(design.adjacency_matrix.sum()) if design.adjacency_matrix is not None else 0,
                )

        context = ExecutionContext(
            task=task,
            topology_strategy=getattr(self.topology_strategy, "strategy_name", ""),
        )
        context.goal = kwargs.get("initial_goal") or self.initial_goal or "Analyze the problem and propose an initial approach."
        context.agents = self.agents
        # Separate state backends:
        # - context.ledger  : legacy linear IllocutionaryLedger (SelfRouting / p2p_free).
        # - context.tlsg_graph: TLSG graph for TlsgRouting and its consumers.
        context.ledger = IllocutionaryLedger()
        if getattr(self.routing_strategy, "uses_tlsg_graph", False):
            from mas_topo.tlsg.graph import TemporalLayeredSpeechActGraph
            context.tlsg_graph = TemporalLayeredSpeechActGraph()
        if kwargs.get("metadata"):
            context.metadata.update(kwargs["metadata"])

        # 初始化每个 Agent 的统一局部邻居表（基于初始拓扑）
        initial_adj, _ = self.topology_strategy.construct_topology(
            self.agent_names, {}, context)
        initial_neighbor_map = self._build_neighbor_map(self.agent_names, initial_adj)
        for agent in self.agents:
            # local_neighbor_table[self.name] 表示当前 Agent 的物理一跳邻居（只读）
            # 其他 key 表示通过交互学习到的其他 Agent 的邻居关系
            agent.local_neighbor_table.setdefault(agent.name, set())
            agent.local_neighbor_table[agent.name] = set(initial_neighbor_map.get(agent.name, []))
        context.neighbor_map = initial_neighbor_map

        result = GraphResult(context=context)

        logger.info("Graph.run: agents=%s max_rounds=%d",
                     self.agent_names, self.max_rounds)

        # ── 预计算常用变量 ──────────────────────────────────────
        manager_name = getattr(self.decision_strategy, 'manager_name', None)
        manager_agent = self.name_to_agent.get(manager_name) if manager_name else None
        skip_manager = (
            manager_name is not None
            and manager_agent is not None
            and getattr(manager_agent, 'llm_adapter', None) is not None
        )
        worker_names = [n for n in self.agent_names if n != manager_name]
        worker_roles = {
            a.name: a.role for a in self.agents if a.name != manager_name
        }

        # 向所有 Agent 注入完整团队名册（名称+角色），供 prompt 构建使用
        context.team_roster = [
            {"name": a.name, "role": a.role}
            for a in self.agents
        ]

        # ── Round 0: Manager 初始推理（仅当路由策略支持选择性激活时）──
        # 路由策略声明 supports_selective_activation=True 时需 Manager 先设定初始目标；
        # 声明 requires_manager_initial=False 的路由（如 AutoGen GroupChat，
        # 原版 Manager 不在 transcript 中产出内容）跳过 Round 0。
        round0_manager_output = None
        supports_selective = getattr(
            self.routing_strategy, "supports_selective_activation", False
        ) and getattr(
            self.routing_strategy, "requires_manager_initial", True
        )
        if supports_selective and skip_manager:
            # Round 0 前注入 team roster，让 Manager 知道可用 workers
            context.team_roster = [
                {"name": a.name, "role": a.role}
                for a in self.agents
            ]
            round0_manager_output = await self._run_manager_initial(task, context)
            if round0_manager_output:
                if round0_manager_output.next_goal:
                    context.goal = round0_manager_output.next_goal
                if round0_manager_output.public_content:
                    manager_agent.append_memory(
                        f"[ROUND 0 - Manager Initial]:\n{round0_manager_output.public_content}"
                    )
                logger.info(
                    "Round 0 Manager initial: next_goal=%s",
                    (context.goal or "")[:80],
                )

        try:
            for round_num in range(1, self.max_rounds + 1):
                t_round_start = time.time()
                context.round_num = round_num
                tokens_before = (context.prompt_tokens, context.completion_tokens)

                if self.verbose:
                    logger.info("Round %d/%d (async)", round_num, self.max_rounds)

                # ── 选择性 Worker 激活 ──────────────────────────────
                # RoutingStrategy 基类默认返回全部 Worker；
                # 子类（如 IllocutionMatcherRouting）可重写以实现选择性激活。
                skip = {manager_name} if skip_manager else set()
                previous_outputs = None
                if result.rounds:
                    previous_outputs = result.rounds[-1].agent_outputs
                elif round0_manager_output:
                    previous_outputs = {manager_name: round0_manager_output}

                active_workers = self.routing_strategy.get_active_workers(
                    context, worker_names,
                    previous_outputs=previous_outputs,
                    worker_roles=worker_roles,
                )
                inactive_workers = set(worker_names) - active_workers
                skip |= inactive_workers
                if self.verbose and inactive_workers:
                    logger.info(
                        "Round %d: active_workers=%s inactive=%s",
                        round_num, sorted(active_workers), sorted(inactive_workers),
                    )

                agent_outputs = await self._run_agents_async(
                    task, context, skip_names=skip)

                # 为未激活 Worker 补充空输出占位，保持 agent_outputs 完整性
                for name in self.agent_names:
                    if name not in agent_outputs:
                        agent_outputs[name] = AgentOutput()

                round_p = context.prompt_tokens - tokens_before[0]
                round_c = context.completion_tokens - tokens_before[1]
                agent_elapsed = {
                    name: out.elapsed_time
                    for name, out in agent_outputs.items()
                    if out.elapsed_time > 0
                }

                adj_matrix, rel_matrix = self.topology_strategy.construct_topology(
                    self.agent_names, agent_outputs, context)
                # 将当前物理邻居表注入上下文，并同步更新每个 Agent 的统一局部邻居表中自身条目
                context.neighbor_map = self._build_neighbor_map(self.agent_names, adj_matrix)
                for agent in self.agents:
                    agent.local_neighbor_table.setdefault(agent.name, set())
                    agent.local_neighbor_table[agent.name] = set(context.neighbor_map.get(agent.name, []))
                aggregation_order = self.topology_strategy.compute_aggregation_order(
                    adj_matrix) if adj_matrix is not None else list(range(len(self.agents)))
                routed_messages = self.routing_strategy.route(
                    self.agent_names, agent_outputs, adj_matrix, context,
                    aggregation_order=aggregation_order)
                self.memory_strategy.update(
                    agents=self.agents,
                    agent_names=self.agent_names,
                    routed_messages=routed_messages,
                    agent_outputs=agent_outputs,
                    round_num=round_num,
                    aggregation_order=aggregation_order,
                    context=context,
                )

                # Phase 5: Manager 决策（论文 Algorithm 1 Line 28-29）
                # S_global ← [C_task; Σ m_pub,i],  Π_meta 输出 y / C_task^(t+1)
                # Manager 以 next_goal 驱动下一轮，无需广播 public_content
                decision = self.decision_strategy.decide(
                    self.agent_names, agent_outputs, context)

                # 将 Phase 5 Manager 输出加入 agent_outputs（仅用于 trace 记录）
                if skip_manager and decision.decision_output is not None:
                    agent_outputs[manager_name] = decision.decision_output

                round_elapsed = time.time() - t_round_start
                context.elapsed_time += round_elapsed

                current_state = self._current_state(context)
                result.rounds.append(RoundResult(
                    round_num=round_num,
                    agent_outputs=agent_outputs,
                    adjacency_matrix=adj_matrix,
                    relevance_matrix=rel_matrix,
                    routed_messages=routed_messages,
                    decision=decision,
                    elapsed_time=round_elapsed,
                    ledger=current_state,
                ))

                if self.topology_logger:
                    self.topology_logger.log_round(
                        round_num=round_num,
                        agent_names=self.agent_names,
                        agent_name_to_role={
                            a.name: getattr(a, "role", "") for a in self.agents
                        },
                        agent_outputs=agent_outputs,
                        adjacency_matrix=adj_matrix,
                        relevance_matrix=rel_matrix,
                        routed_messages=routed_messages,
                        decision_is_complete=decision.is_complete,
                        decision_next_goal=decision.next_goal or "",
                        decision_answer=decision.final_answer,
                        decision_votes=decision.votes,
                        topology_strategy=type(self.topology_strategy).__name__,
                        routing_strategy=type(self.routing_strategy).__name__,
                        task=task,
                        round_goal=context.goal,
                        round_prompt_tokens=round_p,
                        round_completion_tokens=round_c,
                        round_elapsed_time=round_elapsed,
                        agent_elapsed_times=agent_elapsed,
                        ledger_entries=self._serialize_ledger(current_state),
                    )
                    # 同时写入本轮的 LLM 提示词详情
                    self.topology_logger.write_round_prompts(
                        round_num=round_num,
                        prompts=context.get_agent_prompts(),
                        agent_name_to_role={
                            a.name: getattr(a, "role", "") for a in self.agents
                        },
                    )
                    context.clear_agent_prompts()

                if self.verbose:
                    logger.info("Decision: complete=%s", decision.is_complete)

                if decision.is_complete:
                    result.is_completed = True
                    break

                context.goal = decision.next_goal or ""

            # 优先使用决策策略给出的最终答案；决策未产生答案（如 Manager 只裁决
            # is_complete/next_goal）时回退到轮次回溯，从 Solver/Developer 提取
            decision_answer = ""
            if result.is_completed and result.rounds and result.rounds[-1].decision is not None:
                decision_answer = result.rounds[-1].decision.final_answer or ""
            result.final_answer = decision_answer or self._extract_final_solution_from_rounds(result)

            if self.topology_logger:
                self.topology_logger.finalize(
                    result,
                    ledger_entries=self._serialize_ledger(self._current_state(context)),
                )

            logger.info("Graph.run done: completed=%s rounds=%d tokens=%dP/%dC",
                         result.is_completed, len(result.rounds),
                         context.prompt_tokens, context.completion_tokens)
            return result
        finally:
            await self._cleanup_llm_adapters()

    async def _cleanup_llm_adapters(self):
        """清理所有 LLM 适配器的异步客户端连接池。

        必须在事件循环关闭前调用，否则 httpx 连接池的延迟清理回调会
        在 loop 关闭后触发 RuntimeError('Event loop is closed')。
        """
        seen = set()
        # 收集所有 agent 共享的 llm_adapter
        for agent in self.agents:
            adapter = getattr(agent, 'llm_adapter', None)
            if adapter is not None and id(adapter) not in seen:
                seen.add(id(adapter))
                if hasattr(adapter, 'aclose'):
                    try:
                        await adapter.aclose()
                    except Exception:
                        pass
        # decision_strategy 可能持有独立的 llm_adapter
        adapter = getattr(self.decision_strategy, 'llm_adapter', None)
        if adapter is not None and id(adapter) not in seen:
            if hasattr(adapter, 'aclose'):
                try:
                    await adapter.aclose()
                except Exception:
                    pass
        # 让 anyio transport 关闭时通过 call_soon 注册的延迟回调
        # 在 loop.close() 之前有机会执行，避免 Event loop is closed
        try:
            await asyncio.sleep(0)
        except RuntimeError:
            pass

    async def _run_manager_initial(
        self, task: str, context: ExecutionContext
    ) -> Optional[AgentOutput]:
        """Round 0: Manager 初始推理，输出初始状态总结和 next_goal。

        直接复用 LLMAgent.async_execute 的完整推理链路（prompt 构建、
        LLM 调用、JSON 解析、字段校验、重试），避免和 agent.py 重复造轮子。
        """
        manager_name = getattr(self.decision_strategy, 'manager_name', None)
        mgr = self.name_to_agent.get(manager_name) if manager_name else None
        if not mgr or not getattr(mgr, 'llm_adapter', None):
            return None

        # 设置 Round 0 上下文；_build_prompt 会据此拼接初始引导语
        context.round_num = 0
        context.goal = ""
        output = await mgr.async_execute(task, context)

        # Round 0 特殊内容校验（格式校验已由 _validate_output 处理）
        if output:
            missing = []
            if not (output.next_goal or "").strip():
                missing.append("next_goal")
            if output.is_complete:
                missing.append("is_complete (Round 0 MUST be false)")
            if not any(
                re.search(r"illocution\[\w+\]", k, re.IGNORECASE)
                for k in (output.private_content or {}).keys()
            ):
                missing.append("private_content with illocution[Category]")
            if missing:
                logger.warning("Round 0 Manager: missing content %s", missing)

        logger.info(
            "Round 0 Manager initial: is_complete=%s next_goal=%s",
            output.is_complete if output else None,
            (output.next_goal or "")[:80] if output else "",
        )
        return output

    # ── 内部方法 ──────────────────────────────────────────

    @staticmethod
    def _current_state(context: ExecutionContext):
        """Return the active state object for the current routing strategy.

        Prefers ``context.tlsg_graph`` when present (TLSG mode), otherwise
        falls back to ``context.ledger``.
        """
        return getattr(context, "tlsg_graph", None) or getattr(context, "ledger", None)

    def _extract_final_solution_from_rounds(self, result) -> str:
        """从所有轮次中提取最终解，仅从 Agent 的 answer 字段提取。

        按轮次从后往前回溯，优先取 Solver/Developer 的 answer 字段。
        不退回 Manager/decision，确保答案来自求解 Agent。

        若 ledger 中存在对 Developer answer 的验证记录，优先返回
        最新且验证状态为 PASSED 的版本；否则回退到最新 answer。
        """
        if not result.rounds:
            return ""

        ledger = (
            getattr(result.context, "tlsg_graph", None)
            or getattr(result.context, "ledger", None)
        )

        # 1. 优先取 Developer 最新且验证通过的 answer
        if ledger:
            passed_answer = self._find_latest_verified_answer(ledger, "Developer")
            if passed_answer:
                return passed_answer

        # 2. 从所有轮次中优先查找 Solver/Developer 的 answer（从后往前，取最新）
        for round_data in reversed(result.rounds):
            agent_outputs = round_data.agent_outputs
            for preferred in ("Solver", "Developer"):
                out = agent_outputs.get(preferred)
                if out and out.answer and str(out.answer).strip():
                    return str(out.answer).strip()

        # 3. 回退：从所有轮次中取任意 Agent 的 answer
        for round_data in reversed(result.rounds):
            for name, out in round_data.agent_outputs.items():
                if out and out.answer and str(out.answer).strip():
                    return str(out.answer).strip()

        return ""

    @staticmethod
    def _find_latest_verified_answer(ledger, agent_name: str) -> str:
        """从 ledger 中查找指定 Agent 最新且验证通过的 answer_snapshot。

        只返回 verification_status == "PASSED" 的 answer。
        """
        if ledger is None:
            return ""
        verified_by_answer: dict[str, int] = {}
        for entry in ledger.all_entries():
            if (
                entry.answer_snapshot
                and entry.sender == agent_name
                and entry.verification_status == "PASSED"
            ):
                verified_by_answer[entry.answer_snapshot] = max(
                    verified_by_answer.get(entry.answer_snapshot, -1),
                    entry.entry_id,
                )
        if not verified_by_answer:
            return ""
        # 取 entry_id 最大的（即最新的）验证通过 answer
        return max(verified_by_answer.items(), key=lambda x: x[1])[0]

    async def _run_agents_async(
        self, task: str, context: ExecutionContext,
        skip_names: set | None = None,
    ) -> dict[str, AgentOutput]:
        """异步并行执行 Agent。

        Args:
            task: 任务描述。
            context: 执行上下文。
            skip_names: 跳过的 Agent 名称集合（如 Phase 1 跳过 Manager）。
        """
        skip = skip_names or set()

        async def _run_one(agent):
            for attempt in range(self.max_retries):
                try:
                    return agent.name, await agent.async_execute(task, context)
                except Exception as e:
                    actual = e
                    if isinstance(e, RetryError):
                        try:
                            actual = e.last_attempt.exception()
                        except Exception:
                            actual = e
                    logger.warning(
                        "Agent %s attempt %d failed: %s",
                        agent.name, attempt + 1, actual,
                    )
                    if attempt == self.max_retries - 1:
                        return agent.name, AgentOutput(
                            public_content=f"[Error: {actual}]",
                        )

        tasks = [_run_one(a) for a in self.agents if a.name not in skip]
        results = await asyncio.gather(*tasks)
        return dict(results)

    @staticmethod
    def _serialize_ledger(ledger) -> list[dict]:
        """将 ledger 序列化为字典列表（兼容 IllocutionaryLedger 和 TLSG graph）。"""
        entries = []
        for entry in ledger.all_entries():
            entries.append({
                "entry_id": entry.entry_id,
                "round_num": entry.round_num,
                "sender": entry.sender,
                "recipients": entry.recipients,
                "illocution": entry.illocution,
                "content": entry.content[:500],
                "status": entry.status,
                "parent_ids": list(getattr(entry, "parent_ids", [])),
                "answer_snapshot": getattr(entry, "answer_snapshot", None),
                "verification_status": getattr(entry, "verification_status", None),
            })
        return entries

    @staticmethod
    def _build_neighbor_map(
        agent_names: list[str], adjacency_matrix: Optional[np.ndarray]
    ) -> dict[str, list[str]]:
        """根据邻接矩阵构建每个 Agent 的邻居列表。"""
        neighbor_map: dict[str, list[str]] = {name: [] for name in agent_names}
        if adjacency_matrix is None or adjacency_matrix.size == 0:
            return neighbor_map
        n = len(agent_names)
        if adjacency_matrix.shape != (n, n):
            return neighbor_map
        for i, src in enumerate(agent_names):
            for j, dst in enumerate(agent_names):
                if i != j and adjacency_matrix[i][j] == 1:
                    neighbor_map[src].append(dst)
        return neighbor_map

    # ── 工厂方法 ──────────────────────────────────────────

    @classmethod
    def from_config(cls, config, topology_logger=None) -> "Graph":
        """从 FrameworkConfig 构建 Graph 实例。

        Args:
            config: FrameworkConfig dataclass 实例。
            topology_logger: 可选的 TopologyLogger 实例。
                若为 None 且 config.log_dir 已设置，则自动在 log_dir/trace/ 下创建。
        """
        from mas_topo.config.schema import FrameworkConfig
        if not isinstance(config, FrameworkConfig):
            raise TypeError(f"Expected FrameworkConfig, got {type(config)}")

        cls._ensure_registrations()

        # 自动创建 topology_logger + 启用文件日志（若未显式传入且配置了 log_dir）
        if topology_logger is None and config.log_dir:
            from mas_topo.experiment.topology_logger import TopologyLogger
            from mas_topo.utils.logging import setup_file_logging
            setup_file_logging(config.log_dir)
            trace_dir = os.path.join(config.log_dir, "trace")
            topology_logger = TopologyLogger(trace_dir=trace_dir)
            logger.info("Auto-created TopologyLogger at %s", trace_dir)

        from mas_topo import (
            agent_registry, llm_registry,
            topology_registry, decision_registry, routing_registry,
            tool_registry, memory_registry,
        )

        # LLM adapter
        llm_kwargs = {
            "model": config.llm.model,
            "api_key": config.llm.api_key,
            "base_url": config.llm.base_url,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
            "request_timeout": config.llm.request_timeout,
            "json_mode": config.llm.json_mode,
            "disable_thinking": config.llm.disable_thinking,
            "thinking_budget": config.llm.thinking_budget,
            "stream": config.llm.stream,
        }
        llm = llm_registry.get(config.llm.provider, **llm_kwargs)

        # Agents
        agents = []
        for a_cfg in config.agents:
            params = dict(a_cfg.params)
            params.setdefault("llm_adapter", llm)
            params.setdefault("system_prompt", a_cfg.system_prompt)
            params.setdefault("output_format", a_cfg.output_format)
            params.setdefault("output_schema", a_cfg.output_schema)
            params.setdefault("memory_limit", a_cfg.memory_limit)
            params.setdefault("memory_max_rounds", a_cfg.memory_max_rounds)
            params.setdefault("max_tool_rounds", a_cfg.max_tool_rounds)
            params.setdefault("required_output_fields", a_cfg.required_output_fields)
            params.setdefault("context_fields", a_cfg.context_fields)
            params.setdefault("bare_io", a_cfg.bare_io)
            resolved_tools = []
            for tool_name in a_cfg.tools:
                if isinstance(tool_name, str):
                    if not tool_registry.has(tool_name):
                        available = ", ".join(tool_registry.list())
                        raise ValueError(
                            f"Agent '{a_cfg.name}': unknown tool '{tool_name}'. "
                            f"Available tools: {available}"
                        )
                    resolved_tools.append(tool_registry.get(tool_name))
                else:
                    resolved_tools.append(tool_name)
            agent = agent_registry.get(
                a_cfg.type,
                name=a_cfg.name or "",
                role=a_cfg.role or "",
                tools=resolved_tools,
                **params,
            )
            agents.append(agent)

        # Topology strategy — 自动透传所有非 None 字段（避免硬编码列表）
        topo_kw = {
            k: v for k, v in config.topology.model_dump(mode="json", exclude_none=True).items()
            if k != "strategy" and v is not None
        }
        topology_strategy = topology_registry.get(
            config.topology.strategy, **topo_kw)

        # Decision strategy — 自动透传所有非 None 字段
        dec_kw = {
            k: v for k, v in config.decision.model_dump(mode="json", exclude_none=True).items()
            if k != "strategy" and v is not None
        }
        # decision 段未显式设置 max_rounds 时，回退到顶层 max_rounds，
        # 保证决策策略的强制终止/健康门豁免与主循环轮次上限一致
        dec_kw.setdefault("max_rounds", config.max_rounds)
        if "llm_adapter" not in dec_kw:
            dec_kw["llm_adapter"] = llm

        # 将 Manager Agent 实例注入决策策略（论文 Phase 5 二次调用需要）
        manager_name = dec_kw.get("manager_name", "Manager")
        for agent in agents:
            if agent.name == manager_name:
                dec_kw["manager_agent"] = agent
                break

        decision_strategy = decision_registry.get(
            config.decision.strategy, **dec_kw)

        # Routing strategy — 自动透传所有非 None 字段
        routing_kw = {
            k: v for k, v in config.routing.model_dump(mode="json", exclude_none=True).items()
            if k != "strategy" and v is not None
        }
        if config.srac is not None:
            routing_kw.setdefault("srac_config", config.srac)
        # AutoGen GroupChat 原版首个发言人也由 LLM 选择，路由首轮需要 LLM 载体
        if config.routing.strategy == "autogen_groupchat":
            routing_kw.setdefault("llm_adapter", llm)
        routing_strategy = routing_registry.get(
            config.routing.strategy, **routing_kw
        )

        # Memory strategy
        memory_cfg = config.memory or {}
        memory_strategy_name = memory_cfg.get("strategy", "default")
        memory_kw = {
            k: v for k, v in memory_cfg.items()
            if k not in ("strategy", "memory_max_rounds", "memory_limit") and v is not None
        }
        memory_strategy = memory_registry.get(memory_strategy_name, **memory_kw)

        # 全局 memory_max_rounds：若 memory 段配置了则应用到所有 Agent（不覆盖 per-agent 设置）
        global_memory_max_rounds = memory_cfg.get("memory_max_rounds")
        if global_memory_max_rounds is not None:
            for agent in agents:
                if agent.memory_max_rounds is None:
                    agent.memory_max_rounds = global_memory_max_rounds

        # 全局 memory_limit：若 memory 段配置了则覆盖所有 Agent（0 表示不截断）
        global_memory_limit = memory_cfg.get("memory_limit")
        if global_memory_limit is not None:
            for agent in agents:
                agent.memory_limit = global_memory_limit

        return cls(
            agents=agents,
            topology_strategy=topology_strategy,
            decision_strategy=decision_strategy,
            routing_strategy=routing_strategy,
            max_rounds=config.max_rounds,
            verbose=config.verbose,
            topology_logger=topology_logger,
            initial_goal=config.initial_goal,
            memory_strategy=memory_strategy,
        )

    @classmethod
    def from_config_file(cls, path: str) -> "Graph":
        """从 YAML/JSON 配置文件创建 Graph。"""
        from mas_topo.config.loader import load_config
        config = load_config(path)
        return cls.from_config(config)

    @classmethod
    def from_config_dict(cls, data: dict) -> "Graph":
        """从配置字典创建 Graph。"""
        from mas_topo.config.loader import load_config_from_dict
        config = load_config_from_dict(data)
        return cls.from_config(config)

    @staticmethod
    def _ensure_registrations():
        """确保策略模块已导入，触发 @register 装饰器。

        自动扫描 mas_topo 下的 topology、routing、decision、memory 子包，
        替代硬编码模块列表，新增策略无需修改源码。
        """
        import importlib
        import pkgutil
        import mas_topo

        base_pkg = mas_topo.__name__
        subpackages = [
            "topology",
            "routing",
            "decision",
            "memory",
        ]

        # 自动扫描子包下的所有模块
        for subpkg in subpackages:
            pkg_path = f"{base_pkg}.{subpkg}"
            try:
                pkg = importlib.import_module(pkg_path)
                for _finder, name, _ispkg in pkgutil.iter_modules(pkg.__path__, pkg_path + "."):
                    try:
                        importlib.import_module(name)
                    except ImportError:
                        pass
            except ImportError:
                pass

        # 始终导入的基础模块（不在子包扫描范围内）
        core_modules = [
            "mas_topo.agent",
            "mas_topo.llm",
            "mas_topo.tools.builtin",
            "mas_topo.prompt.template",
        ]
        for name in core_modules:
            try:
                importlib.import_module(name)
            except ImportError:
                pass
