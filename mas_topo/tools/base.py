"""工具模块。

提供 Tool 基类和内置工具注册。
"""

from typing import Any
from abc import ABC, abstractmethod


class Tool(ABC):
    """工具抽象基类。

    所有工具必须实现 name、description、parameters 和 run 方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称。"""

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（供 LLM 理解工具用途）。"""

    @property
    def parameters(self) -> dict:
        """工具参数的 JSON Schema 定义。

        子类可覆盖以提供精确的参数 schema。
        默认从 run 方法的类型注解自动推导。
        """
        import inspect
        try:
            sig = inspect.signature(self.run)
        except (ValueError, TypeError):
            return {"type": "object", "properties": {}, "required": []}

        properties = {}
        required = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "agent", "kwargs"):
                continue
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
            param_type = "string"
            if param.annotation is not inspect.Parameter.empty:
                param_type = type_map.get(param.annotation, "string")
            properties[param_name] = {"type": param_type, "description": f"Parameter: {param_name}"}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """执行工具。"""
