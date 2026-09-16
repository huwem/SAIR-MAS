"""通信拓扑详细日志记录器。

将每轮的 Agent 输入/输出、通信 DAG、链路传输内容、路由方式、
计时信息输出为结构化文件（JSON + Markdown）。

过程信息（trace）与结果信息（results）严格分离到不同文件。
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from mas_topo.message import AgentOutput, Message

logger = logging.getLogger(__name__)


@dataclass
class AgentInputRecord:
    """单个 Agent 的输入记录。"""

    agent_name: str = ""
    received_messages: list[dict] = field(default_factory=list)
    task: str = ""
    round_goal: str = ""


@dataclass
class AgentOutputRecord:
    """单个 Agent 的输出记录。"""

    agent_name: str = ""
    role: str = ""
    public_content: str = ""
    private_content: dict = field(default_factory=dict)
    query_descriptor: Optional[str] = None
    key_descriptor: Optional[str] = None
    is_complete: Optional[bool] = None
    next_goal: Optional[str] = None
    answer: Optional[str] = None
    messages: list[dict] = field(default_factory=list)
    private_routing: list[dict] = field(default_factory=list)
    vote_complete: Optional[bool] = None
    tool_calls: list[dict] = field(default_factory=list)
    elapsed_time: float = 0.0


@dataclass
class EdgeRecord:
    """单条通信链路的传输记录。"""

    source: str
    target: str
    content: str
    channel: str = ""
    metadata: dict = field(default_factory=dict)


class TopologyLogger:
    """通信拓扑日志记录器。

    在每轮执行后记录完整的通信拓扑信息：
    - 所有 Agent 的输入和输出
    - 通信链路 DAG（邻接矩阵 + 边列表）
    - 每条链路传输的具体内容
    - 路由决策元数据（通道选择、相似度分数）
    - 每轮和每个 Agent 的执行时间

    过程信息写入 trace.json，结果信息由调用方单独写入 results.json，
    二者严格分离。

    使用方式:
        tlog = TopologyLogger(trace_dir="traces/run_001")
        graph = Graph(..., topology_logger=tlog)
        result = graph.run(task)
        tlog.finalize()
        # 结果由 Runner 写入 trace_dir/../results.json
    """

    def __init__(
        self,
        trace_dir: str = "topology_traces",
        task: str = "",
        max_content_len: int = 10000,
    ):
        self.trace_dir = Path(trace_dir)
        self.task = task
        self.max_content_len = max_content_len
        self._rounds: list[dict] = []
        self._start_time = datetime.now().isoformat()
        self._wall_start = time.time()
        self._truncation_warned: set[str] = set()

        os.makedirs(self.trace_dir, exist_ok=True)

    # ── 内容截断辅助 ──────────────────────────────────────

    def _truncate(self, content: str, label: str = "") -> str:
        """截断过长内容并记录警告。"""
        if len(content) <= self.max_content_len:
            return content
        key = label or content[:40]
        if key not in self._truncation_warned:
            logger.warning(
                "TopologyLogger: content truncated (%d → %d chars) for %s",
                len(content), self.max_content_len, key,
            )
            self._truncation_warned.add(key)
        return content[:self.max_content_len] + f"\n...[truncated, original length: {len(content)}]"

    # ── 主记录方法 ────────────────────────────────────────

    def log_round(
        self,
        round_num: int,
        agent_names: list[str],
        agent_name_to_role: dict[str, str],
        agent_outputs: dict[str, AgentOutput],
        adjacency_matrix: Optional[np.ndarray],
        relevance_matrix: Optional[np.ndarray],
        routed_messages: dict[str, list[Message]],
        decision_is_complete: bool,
        decision_next_goal: str,
        decision_answer: Optional[str],
        topology_strategy: str,
        routing_strategy: str,
        task: str = "",
        decision_votes: Optional[dict[str, bool]] = None,
        round_goal: str = "",
        round_prompt_tokens: int = 0,
        round_completion_tokens: int = 0,
        round_elapsed_time: float = 0.0,
        agent_elapsed_times: Optional[dict[str, float]] = None,
        ledger_entries: Optional[list[dict]] = None,
    ):
        """记录单轮完整的通信拓扑信息。

        Args:
            round_num: 轮次编号。
            agent_names: Agent 名称列表（顺序与邻接矩阵一致）。
            agent_name_to_role: Agent 名称到角色的映射。
            agent_outputs: 各 Agent 的输出。
            adjacency_matrix: 邻接矩阵 (N, N)。
            relevance_matrix: 权重矩阵 (N, N)，无可为 None。
            routed_messages: 路由后的消息（按接收者组织）。
            decision_is_complete: 决策是否完成。
            decision_next_goal: 下一轮目标。
            decision_answer: 最终答案。
            topology_strategy: 拓扑策略名称。
            routing_strategy: 路由策略名称。
            task: 任务描述。
            round_goal: 本轮目标。
            round_prompt_tokens: 本轮 prompt token 消耗。
            round_completion_tokens: 本轮 completion token 消耗。
            round_elapsed_time: 本轮执行时间（秒）。
            agent_elapsed_times: 各 Agent 执行时间。
        """
        N = len(agent_names)
        agent_elapsed = agent_elapsed_times or {}

        # ---- Agent 输入 ----
        agent_inputs: list[AgentInputRecord] = []
        for i, name in enumerate(agent_names):
            msgs = routed_messages.get(name, [])
            agent_inputs.append(AgentInputRecord(
                agent_name=name,
                received_messages=[
                    {
                        "from": m.sender_id,
                        "channel": m.channel,
                        "content": self._truncate(m.content, f"{name}_input_{m.sender_id}"),
                        "content_length": len(m.content),
                        "metadata": {
                            k: v for k, v in m.metadata.items()
                            if k != "content"
                        },
                    }
                    for m in msgs
                ],
                task=task,
                round_goal=round_goal,
            ))

        # ---- Agent 输出 ----
        agent_output_records: list[AgentOutputRecord] = []
        for i, name in enumerate(agent_names):
            out = agent_outputs.get(name)
            if out is None:
                agent_output_records.append(AgentOutputRecord(
                    agent_name=name,
                    role=agent_name_to_role.get(name, ""),
                ))
            else:
                agent_output_records.append(AgentOutputRecord(
                    agent_name=name,
                    role=agent_name_to_role.get(name, ""),
                    public_content=self._truncate(out.public_content, f"{name}_public"),
                    private_content={
                        k: self._truncate(v, f"{name}_private_{k}")
                        for k, v in out.private_content.items()
                    },
                    query_descriptor=out.query_descriptor,
                    key_descriptor=out.key_descriptor,
                    is_complete=out.is_complete,
                    next_goal=out.next_goal,
                    answer=out.answer,
                    messages=[
                        {"to": m.get("to"), "content": self._truncate(m.get("content", ""), f"{name}_msg_{m.get('to', 'unknown')}")}
                        for m in (getattr(out, "messages", []) or [])
                        if isinstance(m, dict)
                    ],
                    private_routing=list(getattr(out, "private_routing", []) or []),
                    vote_complete=getattr(out, "vote_complete", None),
                    tool_calls=list(getattr(out, "tool_calls", []) or []),
                    elapsed_time=round(agent_elapsed.get(name, out.elapsed_time), 3),
                ))

        # ---- 链路 DAG + 路由决策元数据 ----
        # 从实际路由消息 (routed_messages) 构建 edges，而非从 adjacency_matrix 推断。
        # 这确保 FT 路由等策略的通信内容（ft-public-down/ft-public-up/ft-private）
        # 能被正确记录，避免 Manager public_content 被错误显示为发给 Worker 的 private 消息。
        edges: list[EdgeRecord] = []
        routing_metadata: dict[str, dict] = {}  # 路由决策元数据

        for dst_name, msgs in routed_messages.items():
            for msg in msgs:
                src_name = msg.sender_id
                edge_key = f"{src_name}→{dst_name}"

                # 避免重复记录同一 sender→receiver 的 channel 变化
                #（同一轮中同一对 agent 可能有多条消息，如 FT 的 goal + private）
                # 每条消息独立记录，edge_key 加上 channel 区分
                edge_key_unique = f"{edge_key}({msg.channel})"
                if edge_key_unique in routing_metadata:
                    # 追加内容长度统计
                    routing_metadata[edge_key_unique]["content_length"] += len(msg.content)
                    routing_metadata[edge_key_unique]["message_count"] = (
                        routing_metadata[edge_key_unique].get("message_count", 1) + 1
                    )
                else:
                    is_private = "private" in msg.channel
                    is_public = "public" in msg.channel
                    routing_metadata[edge_key_unique] = {
                        "channel": msg.channel,
                        "has_private": is_private,
                        "has_public": is_public,
                        "used_fallback": False,
                        "similarity_score": None,
                        "content_length": len(msg.content),
                        "message_count": 1,
                    }

                edges.append(EdgeRecord(
                    source=src_name,
                    target=dst_name,
                    content=self._truncate(msg.content, f"edge_{src_name}_{dst_name}"),
                    channel=msg.channel,
                    metadata=dict(msg.metadata),
                ))

        # ---- 工具调用自环边（Agent ↔ 自身，视为内部通信） ----
        for name in agent_names:
            out = agent_outputs.get(name)
            if out is None:
                continue
            tool_calls = getattr(out, "tool_calls", []) or []
            for tci, tc in enumerate(tool_calls):
                call = tc.get("call") or {}
                result = tc.get("result") or {}
                tool_name = call.get("name", "unknown")
                args = call.get("arguments", {})
                # 从 result 中提取关键信息
                inner_result = result.get("result") if isinstance(result.get("result"), dict) else result
                passed = inner_result.get("passed") if isinstance(inner_result, dict) else None

                # 构建工具调用摘要
                summary_parts = [f"Tool: {tool_name}"]
                if args:
                    args_preview = json.dumps(args, ensure_ascii=False)
                    if len(args_preview) > 200:
                        args_preview = args_preview[:200] + "..."
                    summary_parts.append(f"Args: {args_preview}")
                if passed is not None:
                    summary_parts.append(f"Result: {'PASSED' if passed else 'FAILED'}")
                elif result.get("success") is False:
                    summary_parts.append(f"Result: ERROR - {result.get('error', 'unknown')}")
                else:
                    summary_parts.append("Result: completed")
                summary = " | ".join(summary_parts)

                edges.append(EdgeRecord(
                    source=name,
                    target=name,
                    content=self._truncate(summary, f"tool_{name}_{tool_name}"),
                    channel="tool",
                    metadata={
                        "illocution_category": "Tool",
                        "tool_name": tool_name,
                        "is_tool_call": True,
                        "passed": passed,
                        "tool_call_index": tci,
                    },
                ))

        # ---- 拓扑构建元数据 ----
        topology_metadata: dict = {
            "num_agents": N,
            "num_edges": len(edges),
            "density": round(len(edges) / (N * (N - 1)), 4) if N > 1 else 0.0,
            "avg_in_degree": round(
                float(adjacency_matrix.sum(axis=0).mean()), 2
            ) if adjacency_matrix is not None else 0.0,
            "avg_out_degree": round(
                float(adjacency_matrix.sum(axis=1).mean()), 2
            ) if adjacency_matrix is not None else 0.0,
        }

        # 如果有关联度矩阵，记录相似度统计
        if relevance_matrix is not None:
            scores = relevance_matrix[relevance_matrix > 0]
            if len(scores) > 0:
                topology_metadata.update({
                    "similarity_mean": round(float(scores.mean()), 4),
                    "similarity_min": round(float(scores.min()), 4),
                    "similarity_max": round(float(scores.max()), 4),
                    "similarity_std": round(float(scores.std()), 4),
                })

        # 构建邻接表
        adj_list: dict[str, list[str]] = {}
        for i, name in enumerate(agent_names):
            adj_list[name] = []
            if adjacency_matrix is not None:
                for j in range(N):
                    if adjacency_matrix[i][j] == 1:
                        adj_list[name].append(agent_names[j])

        round_data = {
            "round_num": round_num,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "round_goal": round_goal,
            "agent_count": N,
            "topology_strategy": topology_strategy,
            "routing_strategy": routing_strategy,
            "adjacency_list": adj_list,
            "edge_count": len(edges),
            "prompt_tokens": round_prompt_tokens,
            "completion_tokens": round_completion_tokens,
            "elapsed_time": round(round_elapsed_time, 3),
            "topology_metadata": topology_metadata,
            "routing_metadata": routing_metadata,
            "ledger": ledger_entries or [],
            "decision": {
                "is_complete": decision_is_complete,
                "next_goal": decision_next_goal,
                "answer": decision_answer,
                "votes": decision_votes or {},
            },
            "agents": {
                name: {
                    "index": i,
                    "role": agent_name_to_role.get(name, ""),
                    "elapsed_time": agent_output_records[i].elapsed_time,
                    "input": {
                        "task": task,
                        "round_goal": round_goal,
                        "received_messages": agent_inputs[i].received_messages,
                    },
                    "output": {
                        "public_content": agent_output_records[i].public_content,
                        "private_content": agent_output_records[i].private_content,
                        "messages": agent_output_records[i].messages,
                        "query_descriptor": agent_output_records[i].query_descriptor,
                        "key_descriptor": agent_output_records[i].key_descriptor,
                        "is_complete": agent_output_records[i].is_complete,
                        "next_goal": agent_output_records[i].next_goal,
                        "answer": agent_output_records[i].answer,
                        "private_routing": agent_output_records[i].private_routing,
                        "vote_complete": agent_output_records[i].vote_complete,
                        "tool_calls": agent_output_records[i].tool_calls,
                    },
                }
                for i, name in enumerate(agent_names)
            },
            "edges": [
                {
                    "source": e.source,
                    "target": e.target,
                    "channel": e.channel,
                    "content": e.content,
                    "content_length": len(e.content),
                    "metadata": e.metadata,
                }
                for e in edges
            ],
        }

        self._rounds.append(round_data)

        # 每轮立即写入 Markdown
        self._write_round_markdown(round_data, edges, agent_inputs,
                                   agent_output_records)

    # ── 输出方法 ──────────────────────────────────────────

    def _write_round_markdown(
        self,
        data: dict,
        edges: list[EdgeRecord],
        agent_inputs: list[AgentInputRecord],
        agent_output_records: list[AgentOutputRecord],
    ):
        """将单轮数据写入 Markdown 文件。"""
        rn = data["round_num"]
        path = self.trace_dir / f"round_{rn:03d}.md"

        lines = []
        lines.append(f"# Round {rn} — Communication Topology Trace\n")
        lines.append(f"**Task:** {data['task']}")
        lines.append(f"**Round goal:** {data['round_goal']}")
        lines.append(f"**Topology strategy:** `{data['topology_strategy']}`")
        lines.append(f"**Routing strategy:** `{data['routing_strategy']}`")
        lines.append(f"**Agents:** {data['agent_count']} | **Edges:** {data['edge_count']}")
        lines.append(f"**Tokens:** {data['prompt_tokens']}P / {data['completion_tokens']}C")
        lines.append(f"**Elapsed time:** {data['elapsed_time']:.2f}s")
        next_goal = data['decision']['next_goal']
        votes = data['decision'].get('votes', {})
        votes_str = ", ".join(f"{k}={v}" for k, v in votes.items()) if votes else "none"
        lines.append(f"**Decision:** complete={data['decision']['is_complete']}, votes={{{votes_str}}}, next_goal={next_goal}\n")

        # ── 计时信息 ──
        lines.append("---\n")
        lines.append("## Timing\n")
        lines.append(f"| Agent | Role | Elapsed |")
        lines.append(f"|-------|------|---------|")
        for i, name in enumerate(data["agents"]):
            agent_data = data["agents"][name]
            et = agent_data.get("elapsed_time", 0)
            lines.append(f"| {name} | {agent_data['role']} | {et:.2f}s |")
        lines.append(f"\n**Round total:** {data['elapsed_time']:.2f}s\n")

        # ── 拓扑元数据 ──
        lines.append("---\n")
        lines.append("## Topology Metadata\n")
        tm = data.get("topology_metadata", {})
        lines.append(f"- Agents: {tm.get('num_agents', '?')}")
        lines.append(f"- Edges: {tm.get('num_edges', '?')}")
        lines.append(f"- Density: {tm.get('density', '?')}")
        lines.append(f"- Avg in-degree: {tm.get('avg_in_degree', '?')}")
        lines.append(f"- Avg out-degree: {tm.get('avg_out_degree', '?')}")
        if "similarity_mean" in tm:
            lines.append(f"- Similarity: mean={tm['similarity_mean']}, min={tm['similarity_min']}, max={tm['similarity_max']}, std={tm['similarity_std']}")
        lines.append("")

        # ── 通信 DAG ──
        lines.append("---\n")
        lines.append("## Communication DAG\n")
        lines.append("```mermaid")
        lines.append("graph LR")
        agent_names = list(data["adjacency_list"].keys())
        for i, name in enumerate(agent_names):
            role = data["agents"][name]["role"]
            safe_name = name.replace(" ", "_")
            lines.append(f'    {safe_name}("{name} [{role}]")')
        seen_edges = set()
        for e in edges:
            key = (e.source, e.target)
            if key in seen_edges:
                continue
            ch = e.channel.split("-")[-1]
            seen_edges.add(key)
            src = e.source.replace(" ", "_")
            dst = e.target.replace(" ", "_")
            lines.append(f"    {src} -- {ch} --> {dst}")
        lines.append("```\n")

        # ── 路由决策详情 ──
        lines.append("---\n")
        lines.append("## Routing Decisions\n")
        rm = data.get("routing_metadata", {})
        if rm:
            lines.append(f"| Edge | Channel | Private | Public | Similarity | Length |")
            lines.append(f"|------|---------|---------|--------|------------|--------|")
            for edge_key, decision in rm.items():
                sim = decision.get("similarity_score", "-")
                lines.append(
                    f"| {edge_key} | {decision['channel']} | "
                    f"{decision['has_private']} | {decision['has_public']} | "
                    f"{sim} | {decision['content_length']} chars |"
                )
            lines.append("")
        else:
            lines.append("*(No routing metadata recorded)*\n")

        # ── 链路传输内容 ──
        lines.append("---\n")
        lines.append("## Link-Level Content\n")
        if edges:
            for i, e in enumerate(edges):
                lines.append(f"### Edge {i+1}: {e.source} → {e.target}")
                lines.append(f"**Channel:** `{e.channel}` | **Length:** {len(e.content)} chars")
                if e.metadata:
                    for mk, mv in e.metadata.items():
                        if mv:
                            lines.append(f"**{mk}:** {str(mv)[:200]}")
                lines.append(f"\n```text\n{e.content}\n```\n")
        else:
            lines.append("*(No edges in this round)*\n")

        # ── Agent 输入 ──
        lines.append("---\n")
        lines.append("## Agent Inputs\n")
        for inp in agent_inputs:
            lines.append(f"### {inp.agent_name}")
            if inp.received_messages:
                for mi, msg in enumerate(inp.received_messages):
                    sender = msg["from"]
                    ch = msg["channel"]
                    clen = msg["content_length"]
                    lines.append(f"- **Msg {mi+1}:** from `{sender}` (ch={ch}, {clen} chars)")
            else:
                lines.append("*(No messages received — initial round)*")
            lines.append("")

        # ── Agent 输出 ──
        lines.append("---\n")
        lines.append("## Agent Outputs\n")
        for out in agent_output_records:
            lines.append(f"### {out.agent_name} ({out.role}) — {out.elapsed_time:.2f}s")
            if out.public_content:
                lines.append(f"**public_content** ({len(out.public_content)} chars):")
                lines.append(f"\n```text\n{out.public_content}\n```\n")
            if out.private_content:
                lines.append("**private_content:**")
                for pk, pv in out.private_content.items():
                    text = str(pv) if not isinstance(pv, str) else pv
                    lines.append(f"- → `{pk}`: {text[:200]}")
                lines.append("")
            if out.query_descriptor:
                lines.append(f"**query_descriptor:** {out.query_descriptor}")
            if out.key_descriptor:
                lines.append(f"**key_descriptor:** {out.key_descriptor}")
            if out.is_complete is not None:
                lines.append(f"**is_complete:** {out.is_complete}")
            if out.next_goal:
                lines.append(f"**next_goal:** {out.next_goal}")
            if out.private_routing:
                lines.append("**private_routing:**")
                for item in out.private_routing:
                    if not isinstance(item, dict):
                        lines.append(f"- *(malformed item)* {str(item)[:200]}")
                        continue
                    rcpts = item.get("recipients", [])
                    illoc = item.get("illocution", "")
                    content = str(item.get("content", ""))[:200]
                    # 兼容新旧字段名：answer_correct（新）/ vote_complete（旧）
                    vote = item.get("answer_correct", item.get("vote_complete"))
                    vote_tag = f" [vote_complete={vote}]" if vote is not None else ""
                    lines.append(f"- → {rcpts} [{illoc}]{vote_tag}: {content}")
            if out.vote_complete is not None:
                lines.append(f"**vote_complete:** {out.vote_complete}")
            if out.answer:
                lines.append(f"**answer:** {out.answer}")
            if out.tool_calls:
                lines.append(f"**tool_calls:** {len(out.tool_calls)}")
                for tci, tc in enumerate(out.tool_calls):
                    call = tc.get("call") or {}
                    result = tc.get("result") or {}
                    inner_result = result.get("result") or {}
                    passed = inner_result.get("passed")
                    lines.append(
                        f"  - [{tci+1}] `{call.get('name', 'unknown')}` "
                        f"passed={passed}"
                    )
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Topology trace written: %s", path)

    def write_round_prompts(
        self,
        round_num: int,
        prompts: dict[str, dict],
        agent_name_to_role: Optional[dict[str, str]] = None,
    ) -> None:
        """将单轮各 Agent 的 LLM 提示词详情写入 Markdown 文件。

        Args:
            round_num: 轮次编号。
            prompts: {agent_name: {system_prompt, user_prompt, messages,
                     prompt_tokens, completion_tokens, elapsed_time, model}}。
            agent_name_to_role: Agent 名称到角色的映射，用于显示。
        """
        if not prompts:
            return

        roles = agent_name_to_role or {}
        path = self.trace_dir / f"prompts_{round_num:03d}.md"

        lines = [f"# Round {round_num} — LLM Prompts\n"]

        for agent_name in sorted(prompts.keys()):
            p = prompts[agent_name]
            role = roles.get(agent_name, "")
            et = p.get("elapsed_time", 0)
            pt = p.get("prompt_tokens", 0)
            ct = p.get("completion_tokens", 0)
            model = p.get("model", "")

            lines.append(f"## {agent_name} ({role}) — {et:.2f}s, {pt}P/{ct}C")
            if model:
                lines.append(f"**Model:** {model}")
            lines.append("")

            lines.append("### System Prompt")
            lines.append("```text")
            lines.append(p.get("system_prompt", ""))
            lines.append("```\n")

            lines.append("### User Prompt")
            lines.append("```text")
            lines.append(p.get("user_prompt", ""))
            lines.append("```\n")

            # 全量 messages（包含 system + user + retry feedback + tool 交互）
            messages = p.get("messages", [])
            if len(messages) > 2:  # 超过基础 system+user 才展示
                lines.append("### Full Message History")
                lines.append(f"({len(messages)} messages including retries/tool interactions)")
                lines.append("```json")
                # 截断过长内容防止文件膨胀
                safe_messages = []
                for m in messages:
                    content = str(m.get("content", ""))
                    if len(content) > 2000:
                        content = content[:2000] + f"\n...[truncated, {len(content)} chars total]"
                    safe_messages.append({"role": m.get("role", ""), "content": content})
                lines.append(json.dumps(safe_messages, ensure_ascii=False, indent=2))
                lines.append("```\n")

            lines.append("---\n")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Prompt trace written: %s", path)

    def finalize(self, graph_result=None, ledger_entries: Optional[list[dict]] = None) -> str:
        """完成日志记录，写入过程追踪 JSON（不含结果信息）。

        过程信息写入 trace.json，结果信息由调用方单独写入。

        Args:
            graph_result: 可选的 GraphResult，用于记录执行状态摘要。

        Returns:
            日志目录路径。
        """
        total_elapsed = time.time() - self._wall_start

        # 过程追踪 JSON（不含最终结果）
        trace_path = self.trace_dir / "trace.json"
        trace_data = {
            "task": self.task,
            "start_time": self._start_time,
            "total_elapsed_time": round(total_elapsed, 3),
            "total_rounds": len(self._rounds),
            "rounds": self._rounds,
            "ledger": ledger_entries or [],
        }
        # 执行状态摘要（轻量，不包含 final_answer / 准确率等结果指标）
        if graph_result:
            trace_data["execution_status"] = {
                "is_completed": graph_result.is_completed,
                "total_rounds": len(graph_result.rounds),
                "total_prompt_tokens": graph_result.context.prompt_tokens,
                "total_completion_tokens": graph_result.context.completion_tokens,
                "total_cost": graph_result.context.cost,
            }
        trace_path.write_text(
            json.dumps(trace_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Process trace JSON written: %s (results excluded)", trace_path)

        # 索引文件
        index_path = self.trace_dir / "index.md"
        lines = [
            f"# Topology Trace Index\n",
            f"**Task:** {self.task[:200]}\n",
            f"**Start time:** {self._start_time}",
            f"**Total elapsed:** {total_elapsed:.2f}s",
            f"**Total rounds:** {len(self._rounds)}\n",
        ]
        if graph_result:
            status = "COMPLETED" if graph_result.is_completed else "TIMEOUT"
            lines.append(f"**Status:** {status}")
            lines.append(f"**Tokens:** {graph_result.context.prompt_tokens}P/{graph_result.context.completion_tokens}C\n")
        lines.append("## Rounds\n")
        for r in self._rounds:
            rn = r["round_num"]
            et = r.get("elapsed_time", 0)
            lines.append(
                f"- [Round {rn}](round_{rn:03d}.md) — "
                f"{r['agent_count']} agents, {r['edge_count']} edges, "
                f"{et:.2f}s, "
                f"topo={r['topology_strategy']}, route={r['routing_strategy']}, "
                f"complete={r['decision']['is_complete']}"
            )
        index_path.write_text("\n".join(lines), encoding="utf-8")

        return str(self.trace_dir)


def write_results_file(results: list[dict], output_path: str) -> str:
    """将实验结果写入独立的结果文件。

    Args:
        results: 结果字典列表，每个字典对应一个任务的结果。
        output_path: 输出文件路径。

    Returns:
        写入的文件路径。
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": datetime.now().isoformat(),
        "total_tasks": len(results),
        "results": results,
    }
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Results written: %s", path)
    return str(path)
