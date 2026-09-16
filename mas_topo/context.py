"""运行时上下文，每次 Graph.run() 创建独立实例，支持并发。"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExecutionContext:
    """运行时上下文，跟踪任务、轮次、LLM 使用量等。"""
    task: str = ""
    goal: str = ""
    round_num: int = 0
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_time: float = 0.0
    metadata: dict = field(default_factory=dict)
    team_roster: list[dict] = field(default_factory=list)
    neighbor_map: dict[str, list[str]] = field(default_factory=dict)
    agents: list[Any] = field(default_factory=list)
    ledger: Optional[Any] = None
    topology_strategy: str = ""

    # Per-round prompt records (cleared at start of each round)
    _agent_prompts: dict[str, dict] = field(default_factory=dict)

    def track_usage(self, cost: float = 0.0, prompt_tokens: int = 0,
                    completion_tokens: int = 0) -> None:
        """累积 LLM 使用量。"""
        self.cost += cost
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def record_agent_prompt(self, agent_name: str, system_prompt: str,
                            user_prompt: str, messages: list[dict],
                            prompt_tokens: int = 0,
                            completion_tokens: int = 0,
                            elapsed_time: float = 0.0,
                            model: str = "") -> None:
        """记录单个 Agent 在本轮中的提示词详情。

        Agent 执行完毕后调用，存储完整 system/user prompt 及最终 messages
        供后续 trace 日志写入。
        """
        self._agent_prompts[agent_name] = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "messages": messages,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed_time": elapsed_time,
            "model": model,
        }

    def get_agent_prompts(self) -> dict[str, dict]:
        """获取本轮所有 Agent 的提示词记录。"""
        return dict(self._agent_prompts)

    def clear_agent_prompts(self) -> None:
        """清空本轮提示词记录（每轮开始时调用）。"""
        self._agent_prompts.clear()
