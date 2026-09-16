"""AutoGen GroupChat 原版（v0.2）风格决策策略。

对照 microsoft/autogen `autogen/agentchat/groupchat.py` 实现两件 Manager 职责：

1. **终止检查**：`is_termination_msg` 语义——发言人消息中含 TERMINATE 令牌
   即终止（原版无 Manager 的 is_complete 裁决）。
2. **发言人选择（speaker_selection_method="auto"）**：旁路 LLM 调用，
   system 消息为 select_speaker_message_template（角色名册 + "只返回角色名"），
   注入完整群聊 transcript，末尾追加 select_speaker_prompt_template；
   对返回文本做提及计数校验，多个/零个名字时用对应模板重问，
   最终失败回退为 last_speaker 的下一个（round-robin）。

Manager 不在共享 transcript 中产生内容——其选择结果只通过
`DecisionResult.decision_output.next_speaker` 传递给路由策略。

模板字符串逐字拷贝自 AutoGen v0.2.35 默认值。
"""

import json
import logging
import re
from typing import Optional

from mas_topo._registries import decision_registry
from mas_topo.context import ExecutionContext
from mas_topo.message import AgentOutput
from mas_topo.decision.base import DecisionStrategy, DecisionResult

logger = logging.getLogger(__name__)

# --- AutoGen v0.2.35 GroupChat 默认模板（逐字） ---
_SELECT_SPEAKER_MESSAGE_TEMPLATE = """You are in a role play game. The following roles are available:
                {roles}.
                Read the following conversation.
                Then select the next role from {agentlist} to play. Only return the role."""

_SELECT_SPEAKER_PROMPT_TEMPLATE = (
    "Read the above conversation. Then select the next role from {agentlist} "
    "to play. Only return the role."
)

_SELECT_SPEAKER_AUTO_MULTIPLE_TEMPLATE = """You provided more than one name in your text, please return just the name of the next speaker. To determine the speaker use these prioritised rules:
    1. If the context refers to themselves as a speaker e.g. "As the..." , choose that speaker's name
    2. If it refers to the "next" speaker name, choose that name
    3. Otherwise, choose the first provided speaker's name in the context
    The names are case-sensitive and should not be abbreviated or changed.
    Respond with ONLY the name of the speaker and DO NOT provide a reason."""

_SELECT_SPEAKER_AUTO_NONE_TEMPLATE = """You didn't choose a speaker. As a reminder, to determine the speaker use these prioritised rules:
    1. If the context refers to themselves as a speaker e.g. "As the..." , choose that speaker's name
    2. If it refers to the "next" speaker name, choose that name
    3. Otherwise, choose the first provided speaker's name in the context
    The names are case-sensitive and should not be abbreviated or changed.
    The only names that are accepted are {agentlist}.
    Respond with ONLY the name of the speaker and DO NOT provide a reason."""


def mentioned_agents(message_content: str, agent_names: list[str]) -> dict:
    """统计文本中各 agent 名字的提及次数（移植 AutoGen `_mentioned_agents`）。

    匹配规则（大小写敏感，词边界）：
    - 精确名字
    - 下划线换成空格（'Story_writer' == 'Story writer'）
    - 下划线转义（'Story_writer' == 'Story\\_writer'）
    """
    mentions = dict()
    for name in agent_names:
        regex = (
            r"(?<=\W)("
            + re.escape(name)
            + r"|"
            + re.escape(name.replace("_", " "))
            + r"|"
            + re.escape(name.replace("_", r"\_"))
            + r")(?=\W)"
        )
        count = len(re.findall(regex, f" {message_content} "))  # 填充以辅助匹配
        if count > 0:
            mentions[name] = count
    return mentions


def _extract_speaker_from_json(content: str, agent_names: list[str]) -> Optional[str]:
    """json_object 模式下从选择器响应中提取角色名。

    接受 {"role": "X"} / {"speaker": "X"} 等字典、纯 JSON 字符串 "X"、
    单元素列表 ["X"]；提取的名字经 mentioned_agents 校验恰好命中一个
    agent 才返回，否则返回 None（调用方回退原版提及计数校验）。
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    candidate: Optional[str] = None
    if isinstance(data, dict):
        lower = {str(k).lower(): v for k, v in data.items()}
        for key in ("role", "speaker", "next_speaker", "agent", "name", "next"):
            if isinstance(lower.get(key), str):
                candidate = lower[key]
                break
        if candidate is None:
            for v in data.values():
                if isinstance(v, str):
                    candidate = v
                    break
    elif isinstance(data, str):
        candidate = data
    elif isinstance(data, list) and data and isinstance(data[0], str):
        candidate = data[0]
    if not candidate:
        return None
    mentions = mentioned_agents(candidate, agent_names)
    if len(mentions) == 1:
        return next(iter(mentions))
    return None


def select_speaker_auto(
    adapter,
    transcript_messages: list[dict],
    roster: list[dict],
    agent_names: list[str],
    last_speaker: Optional[str] = None,
    max_retries: int = 2,
    select_speaker_message_template: str = _SELECT_SPEAKER_MESSAGE_TEMPLATE,
    select_speaker_prompt_template: str = _SELECT_SPEAKER_PROMPT_TEMPLATE,
    multiple_template: str = _SELECT_SPEAKER_AUTO_MULTIPLE_TEMPLATE,
    none_template: str = _SELECT_SPEAKER_AUTO_NONE_TEMPLATE,
    context: Optional[ExecutionContext] = None,
    json_mode: bool = True,
) -> Optional[str]:
    """原版 "auto" 发言人选择（供决策策略与路由首轮选择共用）。

    对齐 AutoGen v0.2 `_auto_select_speaker`：
    - system = select_speaker_message_template（{roles}/{agentlist}），置于消息首位；
    - 其后逐条注入群聊消息（保留各消息原有角色）；
    - 末尾以 role "system"（原版 role_for_select_speaker_messages 默认值）
      追加 select_speaker_prompt_template；
    - 提及计数校验：恰好 1 个名字 → 选中；多个/零个 → 以对应模板重问
      （重问消息同样为 system 角色），最多 1 + max_retries 次尝试；
    - 最终失败 → last_speaker 的下一个（next_agent 语义）。

    偏差（json_mode=True，默认）：选择调用走 response_format=json_object
    结构化输出通道（规避网关对普通文本请求偶发返回心跳空白响应体导致的
    JSONDecodeError），末尾提示追加 JSON 格式说明；解析优先按 JSON 提取
    角色名，失败再回退原版提及计数校验。传 json_mode=False 恢复逐字原版。
    """
    def next_after() -> Optional[str]:
        if not agent_names:
            return None
        if last_speaker in agent_names:
            return agent_names[(agent_names.index(last_speaker) + 1) % len(agent_names)]
        return agent_names[0]

    if adapter is None or not agent_names:
        return next_after()

    roles = "\n".join(
        f"{a['name']}: {a.get('role', '')}".strip()
        for a in (roster or [])
        if a["name"] in agent_names
    )
    agentlist = f"{agent_names}"
    system = select_speaker_message_template.format(roles=roles, agentlist=agentlist)
    final_prompt = select_speaker_prompt_template.format(agentlist=agentlist)
    if json_mode:
        final_prompt += ' Return the role as a JSON object, e.g. {"role": "<name>"}.'
    messages = [
        {"role": "system", "content": system},
        *transcript_messages,
        {
            "role": "system",
            "content": final_prompt,
        },
    ]

    max_attempts = 1 + max_retries
    for attempt in range(max_attempts):
        logger.info("AutoGen select_speaker: attempt %d/%d", attempt + 1, max_attempts)
        try:
            response = adapter.call(messages, json_mode=json_mode)
        except Exception as e:
            # 网关侧故障（如 OpenRouter 心跳空白响应导致 JSONDecodeError）：
            # adapter 内部重试已耗尽，此处按原版兜底语义降级为轮询，
            # 避免单个样本因基础设施抖动而崩溃。
            logger.warning(
                "AutoGen select_speaker: LLM call failed (%s); "
                "falling back to next agent after %s",
                e, last_speaker,
            )
            return next_after()
        usage = getattr(response, "usage", None)
        if usage is not None and context is not None:
            context.track_usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        content = (getattr(response, "content", "") or "").strip()
        if context is not None:
            context.record_agent_prompt(
                "GroupChatManager.select_speaker", system, content, messages,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            )

        mentions = mentioned_agents(content, agent_names)
        if json_mode:
            extracted = _extract_speaker_from_json(content, agent_names)
            if extracted is not None:
                mentions = {extracted: 1}
        if len(mentions) == 1:
            selected = next(iter(mentions))
            logger.info("AutoGen select_speaker: selected %s", selected)
            return selected

        if attempt < max_attempts - 1:
            template = multiple_template if len(mentions) > 1 else none_template
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "system",
                "content": template.format(agentlist=agentlist),
            })

    fallback = next_after()
    logger.warning(
        "AutoGen select_speaker: all %d attempts failed, "
        "falling back to next agent after %s -> %s",
        max_attempts, last_speaker, fallback,
    )
    return fallback


class AutoGenManagerDecision(DecisionStrategy):
    """AutoGen 原版 GroupChatManager 决策策略。

    每轮 decide()：
      a. 终止检查——本轮发言人消息含 termination_keyword 即完成
         （require_worker_answer 守卫：无 worker answer 前忽略，防退化空终止）。
      b. 发言人选择——旁路 LLM 调用，产出下一轮的 next_speaker。
    """

    def __init__(
        self,
        manager_name: str = "Manager",
        manager_agent=None,
        llm_adapter=None,
        termination_keyword: str = "TERMINATE",
        require_worker_answer: bool = True,
        max_selection_retries: int = 2,
        select_speaker_message_template: Optional[str] = None,
        select_speaker_prompt_template: Optional[str] = None,
        select_speaker_auto_multiple_template: Optional[str] = None,
        select_speaker_auto_none_template: Optional[str] = None,
        selector_json_mode: bool = True,
        **kwargs,
    ):
        self.manager_name = manager_name
        self.manager_agent = manager_agent
        self.llm_adapter = llm_adapter
        self.termination_keyword = termination_keyword
        self.require_worker_answer = require_worker_answer
        self.max_selection_retries = max_selection_retries
        self.selector_json_mode = selector_json_mode
        self.select_speaker_message_template = (
            select_speaker_message_template or _SELECT_SPEAKER_MESSAGE_TEMPLATE
        )
        self.select_speaker_prompt_template = (
            select_speaker_prompt_template or _SELECT_SPEAKER_PROMPT_TEMPLATE
        )
        self.select_speaker_auto_multiple_template = (
            select_speaker_auto_multiple_template
            or _SELECT_SPEAKER_AUTO_MULTIPLE_TEMPLATE
        )
        self.select_speaker_auto_none_template = (
            select_speaker_auto_none_template or _SELECT_SPEAKER_AUTO_NONE_TEMPLATE
        )
        self._worker_answer_seen = False
        self._last_worker_answer: Optional[str] = None

    # ── 内部工具 ──────────────────────────────────────────────

    def _adapter(self):
        if self.manager_agent is not None and getattr(
            self.manager_agent, "llm_adapter", None
        ) is not None:
            return self.manager_agent.llm_adapter
        return self.llm_adapter

    def _worker_names(self, agent_names: list[str], context: ExecutionContext) -> list[str]:
        """按配置顺序的 worker 名册（排除 Manager）。"""
        if context.team_roster:
            return [
                a["name"] for a in context.team_roster
                if a["name"] != self.manager_name and a["name"] in agent_names
            ]
        return [n for n in agent_names if n != self.manager_name]

    def _find_speaker(
        self, agent_outputs: dict[str, AgentOutput], worker_names: list[str],
    ) -> Optional[str]:
        """本轮发言人：唯一产生内容的 worker。"""
        for name in worker_names:
            out = agent_outputs.get(name)
            if out and (out.public_content or (out.answer and str(out.answer).strip())):
                return name
        return None

    def _update_worker_answer(
        self, agent_outputs: dict[str, AgentOutput], worker_names: list[str],
    ) -> bool:
        """跨轮锁存 worker answer（守卫只读锁存值）。"""
        for name in worker_names:
            out = agent_outputs.get(name)
            if out and out.answer and str(out.answer).strip():
                self._worker_answer_seen = True
                self._last_worker_answer = str(out.answer).strip()
        return self._worker_answer_seen

    def _next_after(
        self, last_speaker: Optional[str], worker_names: list[str],
    ) -> Optional[str]:
        """原版 next_agent：last_speaker 在名册中的下一个；未知则取第一个。"""
        if not worker_names:
            return None
        if last_speaker in worker_names:
            idx = worker_names.index(last_speaker)
            return worker_names[(idx + 1) % len(worker_names)]
        return worker_names[0]

    def _build_transcript_messages(self, context: ExecutionContext) -> list[dict]:
        """群聊 transcript 消息列表（对齐原版"注入群消息到嵌套选择对话"）：

        任务原文为第一条 user 消息；其后每条广播为独立的 assistant 消息
        （群聊中 agent 产出的消息在原版里角色为 assistant、name 为发言人；
        条目自带 "[From X (channel)]:" 前缀以保留归属）。
        """
        messages = []
        if context.task:
            messages.append({"role": "user", "content": context.task})
        mgr = self.manager_agent
        if mgr is not None and hasattr(mgr, "get_memory_entries"):
            for entry in mgr.get_memory_entries():
                messages.append({"role": "assistant", "content": entry})
        return messages

    def _select_speaker(
        self,
        last_speaker: Optional[str],
        worker_names: list[str],
        context: ExecutionContext,
    ) -> Optional[str]:
        """原版 auto 选择：委托模块级 select_speaker_auto。"""
        return select_speaker_auto(
            self._adapter(),
            self._build_transcript_messages(context),
            context.team_roster,
            worker_names,
            last_speaker=last_speaker,
            max_retries=self.max_selection_retries,
            select_speaker_message_template=self.select_speaker_message_template,
            select_speaker_prompt_template=self.select_speaker_prompt_template,
            multiple_template=self.select_speaker_auto_multiple_template,
            none_template=self.select_speaker_auto_none_template,
            context=context,
            json_mode=self.selector_json_mode,
        )

    # ── 主入口 ────────────────────────────────────────────────

    def decide(
        self,
        agent_names: list[str],
        agent_outputs: dict[str, AgentOutput],
        context: ExecutionContext,
    ) -> DecisionResult:
        worker_names = self._worker_names(agent_names, context)
        worker_answer_available = self._update_worker_answer(
            agent_outputs, worker_names,
        )
        speaker = self._find_speaker(agent_outputs, worker_names)

        # a. 终止检查（is_termination_msg 语义）
        if speaker is not None:
            out = agent_outputs[speaker]
            text = (out.public_content or "") + "\n" + (str(out.answer or ""))
            if self.termination_keyword and self.termination_keyword in text:
                if not self.require_worker_answer or worker_answer_available:
                    logger.info(
                        "AutoGen termination: '%s' found in %s's message.",
                        self.termination_keyword, speaker,
                    )
                    final_answer = (
                        str(out.answer).strip()
                        if out.answer and str(out.answer).strip()
                        else self._last_worker_answer
                    )
                    return DecisionResult(
                        is_complete=True,
                        final_answer=final_answer,
                    )
                logger.info(
                    "AutoGen termination ignored: no worker answer yet "
                    "(require_worker_answer)."
                )

        # b. 发言人选择（为下一轮）
        next_speaker = self._select_speaker(speaker, worker_names, context)
        return DecisionResult(
            is_complete=False,
            decision_output=AgentOutput(next_speaker=next_speaker),
        )


@decision_registry.register(
    "autogen_manager",
    display_name="AutoGen GroupChatManager 决策",
    description="原版 AutoGen：TERMINATE 令牌终止 + LLM 旁路选择下一发言人（提及计数校验/重问/轮询兜底）。",
    config_schema={
        "manager_name": {"type": "str", "default": "Manager", "description": "Manager Agent 名称"},
        "termination_keyword": {"type": "str", "default": "TERMINATE", "description": "is_termination_msg 检测的终止令牌"},
        "require_worker_answer": {"type": "bool", "default": True, "description": "为 True 时，无 Worker 产出 answer 前忽略终止令牌（防退化空终止）"},
        "max_selection_retries": {"type": "int", "default": 2, "description": "发言人选择的最大重问次数（原版默认 2）"},
        "selector_json_mode": {"type": "bool", "default": True, "description": "发言人选择走 json_object 结构化输出（规避网关文本通道故障）；False 恢复原版纯文本"},
    },
)
class AutoGenManagerDecisionRegistered(AutoGenManagerDecision):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
