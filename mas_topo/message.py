"""消息和 Agent 输出数据结构。"""

from dataclasses import dataclass, field
from typing import Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


class AgentOutputSchema(BaseModel):
    """Agent 输出的 Pydantic 验证模型。

    用于解析后结构化验证，确保 LLM 输出的 JSON 字段类型正确。
    同时兼容论文中的空格字段名（如 "public content", "q vector"）
    和代码中的 snake_case /旧字段名字段名。
    """

    model_config = ConfigDict(populate_by_name=True)

    public_content: str = Field(
        default="",
        validation_alias=AliasChoices("public content", "public_content", "publiccontent"),
    )
    private_content: dict[str, str] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("private content", "private_content", "privatecontent"),
    )
    messages: list[dict] = Field(default_factory=list)
    query_descriptor: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("q vector", "q_vector", "qvector", "q desc", "q_desc"),
    )
    key_descriptor: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("k vector", "k_vector", "kvector", "k desc", "k_desc"),
    )
    is_complete: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices("is complete", "is_complete", "iscomplete"),
    )
    next_goal: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("next goal", "next_goal", "nextgoal"),
    )
    next_speaker: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("next speaker", "next_speaker", "nextspeaker"),
    )
    answer: Optional[str] = None
    tool_call: Optional[dict] = None

    # --- PragmaMAS-P2P fields ---
    private_routing: list[dict] = Field(default_factory=list)
    # 新字段名 answer_correct（“确信当前答案正确”）；vote_complete 为旧名，兼容读取。
    vote_complete: Optional[bool] = Field(
        default=None,
        validation_alias=AliasChoices(
            "answer_correct", "answer correct", "answercorrect",
            "vote_complete", "votecomplete",
        ),
    )

    @field_validator("is_complete", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        """容错：将字符串 "true"/"false" 转换为布尔值。"""
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return v

    @field_validator("private_content", mode="before")
    @classmethod
    def coerce_private_content(cls, v):
        """容错：确保 private_content 是 dict[str, str]。"""
        if v is None:
            return {}
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items()}
        return {}

    @field_validator("messages", mode="before")
    @classmethod
    def coerce_messages(cls, v):
        """容错：确保 messages 是 list[dict]。"""
        if v is None:
            return []
        if isinstance(v, list):
            return [m for m in v if isinstance(m, dict) and "to" in m and "content" in m]
        return []

    @field_validator("private_routing", mode="before")
    @classmethod
    def coerce_private_routing(cls, v):
        """容错：确保 private_routing 是 list[dict]，丢弃非 dict 项。

        LLM 偶尔会在数组里混入纯字符串；若不在此过滤，model_validate
        会整体失败并降级到手动提取路径，丢失其他字段的类型容错。
        """
        if v is None:
            return []
        if isinstance(v, list):
            return [item for item in v if isinstance(item, dict)]
        return []

    @field_validator("answer", "next_goal", "query_descriptor", "key_descriptor", mode="before")
    @classmethod
    def coerce_optional_str(cls, v):
        """容错：可选字符串字段若为 None 保持 None，否则强制转字符串。

        防止 LLM 输出 JSON 时把 answer / next_goal 等写成纯数字，
        下游调用 .strip() 或字符串拼接时触发 AttributeError。
        """
        if v is None:
            return None
        return str(v)

    @field_validator("public_content", mode="before")
    @classmethod
    def coerce_public_content(cls, v):
        """容错：public_content 总是字符串。"""
        if v is None:
            return ""
        return str(v)


@dataclass
class Message:
    """Agent 间传递的消息。"""
    content: str
    sender_id: str
    receiver_id: Optional[str] = None      # None 表示广播
    channel: str = "default"
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentOutput:
    """Agent 单轮推理输出。

    支持多种风格：
    - 简单文本：public_content 为自由文本
    - 语义路由：query_descriptor/key_descriptor 用于语义匹配
    - FT (Free Talk)：messages 列表用于显式 to 路由
    - PragmaMAS-P2P：private_routing / vote_complete 用于去中心化语旨路由与共识
    """
    public_content: str = ""
    private_content: dict[str, str] = field(default_factory=dict)
    messages: list[dict] = field(default_factory=list)  # FT: [{"to": str, "content": str}]
    query_descriptor: Optional[str] = None
    key_descriptor: Optional[str] = None
    is_complete: Optional[bool] = None
    next_goal: Optional[str] = None
    next_speaker: Optional[str] = None
    answer: Optional[str] = None
    tool_call: Optional[dict] = None       # {"name": str, "arguments": dict}
    tool_calls: list[dict] = field(default_factory=list)  # 历史工具调用 [{call, result}, ...]
    raw_response: Optional[str] = None
    elapsed_time: float = 0.0
    metadata: Optional[dict] = None

    # --- PragmaMAS-P2P fields ---
    private_routing: list[dict] = field(default_factory=list)
    vote_complete: Optional[bool] = None

    def __post_init__(self):
        """从 metadata 中解析 P2P 字段，兼容基于 metadata 的构造方式。"""
        # 容错：字符串字段应为字符串，防止 LLM 输出纯数字后
        # 下游调用 .strip() 或做字符串拼接时触发 AttributeError。
        for field_name in ("answer", "public_content", "next_goal",
                           "query_descriptor", "key_descriptor", "next_speaker"):
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, str):
                setattr(self, field_name, str(value))

        if self.metadata:
            if "private_routing" in self.metadata and not self.private_routing:
                self.private_routing = list(self.metadata["private_routing"])
            for key in ("answer_correct", "vote_complete"):
                if key in self.metadata and self.vote_complete is None:
                    self.vote_complete = bool(self.metadata[key])

    @property
    def has_semantic_descriptors(self) -> bool:
        return self.query_descriptor is not None and self.key_descriptor is not None

    @staticmethod
    def from_text(text: str) -> "AgentOutput":
        """从纯文本创建。"""
        return AgentOutput(public_content=text, raw_response=text)

    def to_text(self) -> str:
        """提取主要文本内容，优先返回 answer。"""
        text = self.answer or self.public_content
        return str(text) if text is not None else ""


class Channel:
    """通信信道常量。"""

    BROADCAST = "ft-broadcast"
    FT_PRIVATE = "ft-private"
    SEMANTIC_PRIVATE = "semantic-private"
    SEMANTIC = "semantic"
    SPATIAL = "spatial"
    FT_PUBLIC_UP = "ft-public-up"
