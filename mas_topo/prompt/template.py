"""模板 PromptProvider。

从配置文件加载角色和 prompt 模板。
"""

import json
from pathlib import Path
from typing import Optional

from mas_topo._registries import prompt_registry
from mas_topo.prompt.base import PromptProvider


class TemplatePromptProvider(PromptProvider):
    """从配置字典加载角色和 prompt 模板的 PromptProvider。

    配置格式:
        roles:
          Manager:
            system_prompt: "You are the Manager..."
            description: "Workflow orchestrator"
          Developer:
            system_prompt: "You are the Developer..."
        templates:
          answer: "Answer the following: {question}"
    """

    def __init__(
        self,
        roles: Optional[dict[str, dict]] = None,
        templates: Optional[dict[str, str]] = None,
    ):
        self._roles: dict[str, dict] = roles or {}
        self._templates: dict[str, str] = templates or {}

    @classmethod
    def from_file(cls, path: str) -> "TemplatePromptProvider":
        """从 YAML/JSON 文件加载。"""
        p = Path(path)
        text = p.read_text(encoding="utf-8")

        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml
                data = yaml.safe_load(text) or {}
            except ImportError:
                raise ImportError("PyYAML required: pip install pyyaml")
        elif p.suffix == ".json":
            data = json.loads(text)
        else:
            raise ValueError(f"Unsupported format: {p.suffix}")

        return cls(
            roles=data.get("roles", {}),
            templates=data.get("templates", {}),
        )

    def get_system_prompt(self, role: str) -> str:
        role_data = self._roles.get(role, {})
        return role_data.get("system_prompt", f"You are {role}.")

    def get_roles(self) -> list[str]:
        return list(self._roles.keys())

    def get_role_description(self, role: str) -> str:
        role_data = self._roles.get(role, {})
        return role_data.get("description", self.get_system_prompt(role))

    def get_prompt(self, prompt_type: str, **kwargs) -> Optional[str]:
        template = self._templates.get(prompt_type)
        if template is None:
            return None
        try:
            return template.format(**kwargs)
        except KeyError:
            return template


@prompt_registry.register("template")
class TemplatePromptProviderRegistered(TemplatePromptProvider):
    def __init__(self, roles=None, templates=None, **kwargs):
        super().__init__(roles=roles, templates=templates)
