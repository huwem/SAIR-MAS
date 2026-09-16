"""Manager Agent 决策策略。

实现论文 Algorithm 1 Phase 5：聚合所有 agent 的 public_content 为全局状态，
再次调用 Manager LLM 决定 is_complete/next_goal。

若 manager_agent 未注入（向后兼容），则回退为读取 Phase 1 输出。

论文 Algorithm 1 Line 28-29:
  S_global^(t) ← [C_task^(t); Σ m_pub,i^(t)]
  ⟨y, C_task^(t+1)⟩ ∼ Π_meta(· | S_global^(t))

Manager 只输出 is_complete (y) 和 next_goal (C_task^(t+1))，
最终解由系统从 S_global 中提取（Algorithm 1 Line 32）。
"""

import json
import logging
from typing import Optional

from mas_topo._registries import decision_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.decision.base import DecisionStrategy, DecisionResult

logger = logging.getLogger(__name__)

# Phase 5 Manager 决策用户消息模板。
# 论文 Algorithm 1 Line 25-26:
#   S(t)_global ← [C(t)_task; Σ_σ(t)({m(t)_pub,i | ai ∈ A})]
#   ⟨y, C(t+1)_task⟩ ∼ Π_meta(· | S(t)_global)
#
# 论文未定义独立的 Phase 5 prompt —— Manager 复用 Phase 1 的 system prompt (Π_meta)，
# 用户消息仅为 S_global：task 拼接所有 agent 的 public 输出。
_PHASE5_USER_PROMPT_TEMPLATE = """{task}

{all_public_content}"""


class ManagerDecision(DecisionStrategy):
    """Manager Agent 决策策略。

    论文 Phase 5：调用 Manager LLM，输入 S_global = [C_task; Σ m_pub,i]，
    输出 ⟨is_complete, next_goal⟩。

    当 manager_agent 未注入时，回退为从 Phase 1 输出读取。
    """

    def __init__(self, manager_name: str = "Manager", manager_agent=None, phase5_prompt_template: Optional[str] = None, require_worker_answer: bool = False):
        self.manager_name = manager_name
        self.manager_agent = manager_agent
        self.phase5_prompt_template = phase5_prompt_template
        self.require_worker_answer = require_worker_answer
        self._worker_answer_seen = False

    def _has_worker_answer(self, agent_names: list[str], agent_outputs: dict[str, AgentOutput]) -> bool:
        """是否有 Worker 产出过 answer（跨轮记忆，本轮命中即锁存）。

        require_worker_answer=True 时用于拦截 Manager 抢跑：没有任何 Worker
        产出过最终答案之前，Manager 的 is_complete 无效——防止 Manager 自己
        解题并首轮终止，使中心化群聊退化为单智能体。
        """
        for name in agent_names:
            if name == self.manager_name:
                continue
            out = agent_outputs.get(name)
            if out and out.answer and str(out.answer).strip():
                self._worker_answer_seen = True
                return True
        return self._worker_answer_seen

    def _detect_answer_convergence(
        self,
        agent_outputs: dict[str, AgentOutput],
    ) -> tuple[bool, str]:
        """检测所有 Agent 是否已收敛到一致答案。

        Returns:
            (是否收敛, 收敛的答案)
        """
        answers = []
        for out in agent_outputs.values():
            ans = out.answer
            if ans is not None and str(ans).strip():
                answers.append(str(ans).strip())

        if len(answers) < 2:
            return False, ""

        first = answers[0]
        if all(a == first for a in answers):
            return True, first

        return False, ""

    @staticmethod
    def _validate_phase5_output(output: AgentOutput) -> list[str]:
        """验证 Phase 5 Manager 输出是否包含必要字段。

        Returns:
            缺失或为空的字段名列表。
        """
        missing = []
        if output.is_complete is None:
            missing.append("is_complete")
        if not output.next_goal or not str(output.next_goal).strip():
            missing.append("next_goal")
        # public_content 是可选字段（用于 PragmaMAS 状态总结，不影响核心决策）
        return missing

    def _call_manager_phase5(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> AgentOutput:
        """执行论文 Phase 5 Manager 二次调用。

        聚合所有 agent 的 public_content，构造 S_global 输入，
        调用 Manager LLM 获取 is_complete / next_goal。
        若输出格式不符合要求，自动重试（最多 2 次）。
        """
        mgr = self.manager_agent
        if mgr is None or mgr.llm_adapter is None:
            return None

        # 构建 system prompt（避免与论文 prompt 重复追加 schema）
        system = mgr.system_prompt
        _system_lower = system.lower()
        _has_json_format = (
            "strict response format" in _system_lower
            and "valid json object" in _system_lower
        )
        if mgr.output_format == "json" and mgr.output_schema and not _has_json_format:
            schema_str = json.dumps(mgr.output_schema, indent=2)
            system += (
                f"\n\nYou MUST output ONLY a valid JSON object "
                f"with these fields:\n{schema_str}"
            )

        # 论文格式 prompt：在 system 最前面追加格式强制声明
        if _has_json_format:
            system = (
                "CRITICAL OUTPUT CONSTRAINT: "
                "Your ENTIRE response must be a single valid JSON object. "
                "Do NOT output markdown, code blocks, explanations, or any text "
                "outside the JSON. Output ONLY the JSON, starting with { and "
                "ending with }.\n\n"
            ) + system

        # 聚合所有 agent 的 public_content + answer = S_global（论文 Line 28）
        parts = []
        for name in agent_names:
            out = agent_outputs.get(name)
            if out:
                content_parts = []
                if out.public_content:
                    content_parts.append(out.public_content)
                if out.answer and str(out.answer).strip():
                    content_parts.append(f"[Answer]:\n{out.answer}")
                if content_parts:
                    parts.append(f"--- {name} ---\n" + "\n".join(content_parts))
            elif name == self.manager_name:
                parts.append(
                    f"--- {name} ---\n"
                    f"[No output this round — the manager should "
                    f"review all outputs below and make a decision.]"
                )
        all_public = "\n\n".join(parts) if parts else "(No agent outputs this round)"

        template = self.phase5_prompt_template or _PHASE5_USER_PROMPT_TEMPLATE
        user = template.format(
            task=context.task,
            all_public_content=all_public,
        )

        # 论文格式时在 user prompt 末尾追加格式提醒
        if _has_json_format:
            user += "\n\nRemember: output ONLY the JSON object, no other text."

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        output = None
        last_response = ""
        last_usage = None
        max_retries = 2

        for attempt in range(max_retries + 1):
            logger.info(
                "Phase 5: calling Manager '%s' for global decision (attempt %d/%d)",
                self.manager_name, attempt + 1, max_retries + 1,
            )
            llm_response = mgr.llm_adapter.call(messages)
            last_response = llm_response.content
            last_usage = llm_response.usage

            output = mgr._parse_json_response(last_response)

            missing = self._validate_phase5_output(output)
            if not missing:
                break

            if attempt < max_retries:
                logger.warning(
                    "Phase 5 Manager: missing fields %s, retrying...", missing,
                )
                missing_str = ", ".join(missing)
                feedback = (
                    f"Your previous response was invalid: "
                    f"missing required fields [{missing_str}]. "
                    f"Please output a valid JSON object with ALL required fields. "
                    f"Output ONLY the JSON, no markdown, no explanations."
                )
                messages.append({"role": "user", "content": feedback})

        context.track_usage(
            prompt_tokens=last_usage.prompt_tokens,
            completion_tokens=last_usage.completion_tokens,
        )

        logger.info(
            "Phase 5 Manager decision: is_complete=%s next_goal=%s",
            output.is_complete if output else None,
            (output.next_goal or "")[:80] if output else "",
        )

        return output

    def decide(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> DecisionResult:
        # 每轮无条件扫描，维护 answer 跨轮锁存（守卫只读锁存值，
        # 否则 is_complete=False 的轮次不会更新锁存）
        worker_answer_available = self._has_worker_answer(agent_names, agent_outputs)

        # 论文 Phase 5: 聚合全局状态，再次调用 Manager
        phase5_out = self._call_manager_phase5(agent_names, agent_outputs, context)

        if phase5_out is not None:
            is_complete = phase5_out.is_complete or False
            next_goal = phase5_out.next_goal or "Continue working."
            if is_complete and self.require_worker_answer and not worker_answer_available:
                logger.info(
                    "Phase 5 Manager: is_complete ignored "
                    "(require_worker_answer, no worker answer yet)."
                )
                is_complete = False
                next_goal = "No worker has produced a final answer yet. Continue working."
            # 论文 Algorithm 1 Line 32: 最终解由 Graph._extract_final_solution 从
            # 全球状态 S_global 中提取。Manager 只裁决 is_complete/next_goal，
            # final_answer 置 None，由 Graph 回退到轮次回溯（若误用
            # public_content 散文总结，代码任务评测会全部 "no code found"）。
            return DecisionResult(
                is_complete=is_complete,
                next_goal=next_goal,
                final_answer=None,
                decision_output=phase5_out,
            )

        # 回退：manager_agent 未注入时，从 Phase 1 输出读取
        manager_out = agent_outputs.get(self.manager_name)
        if manager_out is None:
            return DecisionResult(is_complete=False, next_goal="Continue working.")

        is_complete = manager_out.is_complete or False
        next_goal = manager_out.next_goal or "Continue working."
        if is_complete and self.require_worker_answer and not worker_answer_available:
            is_complete = False
            next_goal = "No worker has produced a final answer yet. Continue working."
        final_answer = manager_out.answer or manager_out.public_content

        return DecisionResult(
            is_complete=is_complete,
            next_goal=next_goal,
            final_answer=final_answer,
            decision_output=manager_out,
        )


@decision_registry.register(
    "manager",
    display_name="Manager 决策",
    description="从指定 Manager Agent 输出中读取 is_complete/next_goal/answer",
    config_schema={
        "manager_name": {"type": "str", "default": "Manager", "description": "Manager Agent 名称"},
        "require_worker_answer": {"type": "bool", "default": False, "description": "为 True 时，无 Worker 产出 answer 前忽略 Manager 的 is_complete（防首轮抢跑）"},
    },
)
class ManagerDecisionRegistered(ManagerDecision):
    def __init__(self, manager_name: str = "Manager", manager_agent=None, phase5_prompt_template: Optional[str] = None, require_worker_answer: bool = False, **kwargs):
        super().__init__(manager_name=manager_name, manager_agent=manager_agent, phase5_prompt_template=phase5_prompt_template, require_worker_answer=require_worker_answer)
