"""Task-conditioned system designer 抽象接口。

在 Graph 执行循环之前，根据 task 动态设计多智能体系统：
- 确定需要哪些 Agent（数量、角色、system_prompt）
- 生成初始通信拓扑（邻接矩阵）

对应 ARG-Designer 论文的自回归图生成思想：将多智能体系统设计
视为条件自回归图生成任务，为每个 task 自动确定最优 agent 组合和通信结构。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TaskDesign:
    """Task-conditioned 系统设计结果。

    Attributes:
        agent_configs: 动态确定的 Agent 配置列表（名称、角色、system_prompt）。
        adjacency_matrix: 初始通信拓扑邻接矩阵。若为 None，由 topology_strategy 每轮动态构建。
        description: 设计说明（用于日志）。
    """
    agent_configs: list = field(default_factory=list)
    adjacency_matrix: Optional[np.ndarray] = None
    description: str = ""


class TaskDesigner(ABC):
    """Task-conditioned 系统设计器抽象基类。

    在 Graph.run() 的 pre-execution 阶段被调用，分析 task 并返回
    TaskDesign（包含动态 Agent 列表和初始拓扑）。

    子类只需实现 design() 方法。

    用法:
        designer = ARGTopologyDesigner(llm=llm, role_pool=[...])
        graph = Graph(agents=[], topology_strategy=..., task_designer=designer)
        result = graph.run("Solve this math problem...")
        # designer.design() 会在 pre-execution 阶段自动调用
    """

    @abstractmethod
    def design(self, task: str) -> TaskDesign:
        """根据 task 设计多智能体系统。

        Args:
            task: 任务描述。

        Returns:
            TaskDesign: 包含动态 Agent 配置列表和初始邻接矩阵。
        """
        ...
