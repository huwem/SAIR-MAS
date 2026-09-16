"""Prompt 模块。

提供 PromptProvider 基类和模板实现。
"""

from abc import ABC, abstractmethod
from typing import Optional


class PromptProvider(ABC):
    """Prompt 提供者抽象基类。

    按角色提供 system_prompt，按类型获取 prompt 模板。
    """

    @abstractmethod
    def get_system_prompt(self, role: str) -> str:
        """获取指定角色的 system prompt。"""

    @abstractmethod
    def get_roles(self) -> list[str]:
        """获取所有可用角色名。"""

    def get_role_description(self, role: str) -> str:
        """获取角色描述（默认实现：直接返回 system_prompt）。"""
        return self.get_system_prompt(role)

    def get_prompt(self, prompt_type: str, **kwargs) -> Optional[str]:
        """按类型获取 prompt 模板，子类可覆盖以扩展。"""
        return None
