"""通用注册表，通过装饰器注册类，按名称获取实例。

支持组件元数据注册，为图形化界面提供策略发现和参数描述能力。
"""

from __future__ import annotations

from typing import Type, Any, Callable, Optional


class Registry:
    """通用注册表，替代分散的 AgentRegistry/LLMRegistry/PromptSetRegistry。

    使用装饰器模式注册类，通过名称获取实例或类。
    支持注册时附带元数据（config_schema、display_name、description 等），
    供 GUI 自动发现策略参数和生成配置表单。
    """

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, Type] = {}
        self._metadata: dict[str, dict] = {}

    def register(
        self,
        name: str,
        *,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        config_schema: Optional[dict] = None,
        category: Optional[str] = None,
    ) -> Callable:
        """装饰器，将类注册到注册表并附带元数据。

        Args:
            name: 注册标识名。
            display_name: 显示名称（GUI 用）。
            description: 描述文本（GUI 用）。
            config_schema: 构造参数字段定义，格式为
                {"field_name": {"type": "str", "default": "...", "description": "..."}}。
            category: 分类标签（GUI 用）。
        """
        def decorator(cls: Type) -> Type:
            if name in self._registry:
                raise ValueError(
                    f"{self.name} registry: '{name}' already registered"
                )
            self._registry[name] = cls
            self._metadata[name] = {
                "display_name": display_name or name,
                "description": description or "",
                "config_schema": config_schema or {},
                "category": category or "",
            }
            return cls
        return decorator

    def get(self, key: str, **kwargs) -> Any:
        """根据注册名获取类的实例，kwargs 传递给构造函数。"""
        if key not in self._registry:
            available = ", ".join(self._registry.keys()) or "(none)"
            raise KeyError(
                f"{self.name} registry: '{key}' not found. "
                f"Available: {available}"
            )
        return self._registry[key](**kwargs)

    def get_class(self, name: str) -> Type:
        """根据名称获取注册类本身（不实例化）。"""
        if name not in self._registry:
            raise KeyError(f"{self.name} registry: '{name}' not found")
        return self._registry[name]

    def has(self, name: str) -> bool:
        return name in self._registry

    def list(self) -> list[str]:
        return list(self._registry.keys())

    def metadata(self, name: str) -> dict:
        """获取注册项的元数据。"""
        if name not in self._metadata:
            raise KeyError(f"{self.name} registry: '{name}' not found")
        return dict(self._metadata[name])

    def list_with_metadata(self) -> list[dict]:
        """列出所有注册项及其元数据，供 GUI 发现组件。"""
        return [
            {
                "name": name,
                **self._metadata[name],
            }
            for name in self._registry.keys()
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        names = ", ".join(self._registry.keys())
        return f"Registry({self.name}, [{names}])"
