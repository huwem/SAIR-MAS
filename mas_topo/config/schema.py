"""配置数据模型定义。

使用 Pydantic BaseModel 提供自动类型验证、JSON Schema 生成和序列化能力，
为图形化界面和配置文件驱动提供基础设施。
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class AgentConfig(BaseModel):
    """Agent 配置。"""

    type: str = Field(default="LLMAgent", description="Agent 类型名，对应 agent_registry 注册名")
    name: Optional[str] = Field(default=None, description="Agent 名称，唯一标识")
    role: Optional[str] = Field(default=None, description="Agent 角色描述")
    tools: list[str] = Field(default_factory=list, description="工具名称列表，对应 tool_registry 注册名")
    memory_limit: int = Field(default=3000, ge=100, description="Agent 记忆历史截断长度")
    memory_max_rounds: Optional[int] = Field(default=None, ge=1, description="仅保留最近 N 轮记忆（None 表示不限制）")
    max_tool_rounds: Optional[int] = Field(default=None, ge=1, description="Agent 每轮执行最多可调用的工具轮数（None 表示使用框架默认值）")
    params: dict = Field(default_factory=dict, description="额外构造参数")
    neighbors: list[dict] = Field(default_factory=list, description="邻居 Agent 列表，每项含 name 和 capability")
    system_prompt: Optional[str] = Field(default=None, description="系统提示词，覆盖默认角色提示")
    output_format: str = Field(default="text", description="输出格式: text | json")
    output_schema: Optional[dict] = Field(default=None, description="JSON 输出字段约束定义")
    required_output_fields: list[str] = Field(
        default_factory=list,
        description="Agent 输出必填字段名列表（如 public_content, q_vector, k_vector, messages, answer 等）。"
                    "空列表表示不强制验证。字段名映射: q_vector→query_descriptor, k_vector→key_descriptor。",
    )
    context_fields: list[str] = Field(
        default_factory=list,
        description="运行时上下文 metadata 中的字段名列表，仅对该 Agent 追加到用户 prompt。"
                    "用于在不泄露给所有 Agent 的情况下给特定 Agent 注入额外信息（如测试用例）。",
    )
    bare_io: bool = Field(
        default=False,
        description="裸调用模式：无 system prompt，user prompt 即题目原文，输出全文直接作为 answer（direct 基线用）",
    )


class TopologyConfig(BaseModel):
    """拓扑策略配置。"""

    strategy: str = Field(default="semantic", description="拓扑策略名，对应 topology_registry 注册名")
    pattern: Optional[str] = Field(default=None, description="fixed 模式专用拓扑模式")
    threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="语义相似度阈值")
    max_in_degree: int = Field(default=3, ge=1, description="最大入度约束")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="嵌入模型名称")
    adjacency: Optional[list[list[int]]] = Field(default=None, description="manual 模式专用邻接矩阵")
    agent_neighbor_map: Optional[dict[str, list[str]]] = Field(
        default=None,
        description="neighbor 模式专用显式邻居映射 {agent_name: [neighbor_name, ...]}",
    )
    seed: Optional[int] = Field(default=None, description="fixed random 模式专用随机种子")
    manager_name: Optional[str] = Field(default=None, description="Manager Agent 名称，启用后拓扑仅计算 Worker 间连接")

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        valid_patterns = {
            "full_connected", "chain", "star", "ring",
            "tree", "mesh", "layered", "random", "direct_answer",
        }
        if v not in valid_patterns:
            raise ValueError(f"Invalid pattern '{v}'. Valid: {sorted(valid_patterns)}")
        return v


class DecisionConfig(BaseModel):
    """决策策略配置。"""

    strategy: str = Field(default="manager", description="决策策略名，对应 decision_registry 注册名")
    manager_name: str = Field(default="Manager", description="manager 策略专用 Manager Agent 名称")
    system_prompt: Optional[str] = Field(default=None, description="自定义决策提示词")
    max_rounds: Optional[int] = Field(default=None, ge=1, description="最大执行轮次（None 时回退到顶层 max_rounds）")
    phase5_prompt_template: Optional[str] = Field(
        default=None,
        description="ManagerDecision Phase 5 prompt 模板（覆盖默认模板）",
    )
    require_worker_answer: Optional[bool] = Field(
        default=None,
        description="manager/autogen_manager 策略专用：为 True 时无 Worker 产出 answer 前忽略终止信号（防首轮抢跑退化为单智能体/空终止）",
    )

    # --- AutoGen GroupChatManager (autogen_manager) fields ---
    termination_keyword: Optional[str] = Field(
        default=None,
        description="autogen_manager 策略专用：is_termination_msg 检测的终止令牌（默认 TERMINATE）",
    )
    max_selection_retries: Optional[int] = Field(
        default=None, ge=0,
        description="autogen_manager 策略专用：发言人选择的最大重问次数（原版默认 2）",
    )
    select_speaker_message_template: Optional[str] = Field(
        default=None, description="autogen_manager 策略专用：覆盖原版 select_speaker_message_template",
    )
    select_speaker_prompt_template: Optional[str] = Field(
        default=None, description="autogen_manager 策略专用：覆盖原版 select_speaker_prompt_template",
    )
    select_speaker_auto_multiple_template: Optional[str] = Field(
        default=None, description="autogen_manager 策略专用：覆盖原版多名重问模板",
    )
    select_speaker_auto_none_template: Optional[str] = Field(
        default=None, description="autogen_manager 策略专用：覆盖无名重问模板",
    )
    selector_json_mode: Optional[bool] = Field(
        default=None,
        description="autogen_manager 策略专用：发言人选择走 json_object 结构化输出（默认 True）；False 恢复原版纯文本选择",
    )

    # --- PragmaMAS-P2P consensus fields ---
    required_answer_role: Optional[str] = Field(default=None, description="必须有非空 answer 产出才算完成的角色")
    required_true_voters: Optional[list[str]] = Field(default=None, description="必须投 True 才允许终止的角色名单（如验证角色 Tester/Verifier）")
    min_health: float = Field(default=0.3, ge=0.0, le=1.0, description="TLSG 言后效果收敛度最低阈值（CLOSED 节点比例），低于此值阻止终止")


class RoutingConfig(BaseModel):
    """路由策略配置。"""

    model_config = ConfigDict(extra="allow")

    strategy: str = Field(default="semantic", description="路由策略名，对应 routing_registry 注册名")
    memory_truncate: int = Field(default=500, ge=0, description="消息内容截断长度，0 表示不截断")
    manager_name: Optional[str] = Field(default=None, description="Manager Agent 名称，启用后分离 public/private 信道")
    first_round_exclude: Optional[list[str]] = Field(default=None, description="首轮不激活的 Agent 名单（如验证角色 Tester/Verifier），None 表示首轮全激活")


class LLMConfig(BaseModel):
    """LLM 配置。"""

    provider: str = Field(default="openai_compat", description="LLM 提供商，对应 llm_registry 注册名")
    model: str = Field(default="gpt-4o", description="模型名称")
    api_key: Optional[str] = Field(default=None, description="API 密钥")
    base_url: Optional[str] = Field(default=None, description="API 基础 URL")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="采样温度")
    max_tokens: int = Field(default=16000, ge=1, description="最大生成 token 数")
    request_timeout: float = Field(default=900.0, ge=0.0, description="单次 LLM 请求总墙钟超时（秒），0 表示不限制")
    json_mode: bool = Field(default=False, description="启用 OpenAI JSON mode（response_format）")
    disable_thinking: bool = Field(default=False, description="禁用 DeepSeek 等推理模型的 thinking/reasoning 阶段")
    thinking_budget: int = Field(default=0, ge=0, description="thinking/reasoning 阶段的 token 预算上限，0 表示不限制")
    stream: bool = Field(default=False, description="流式调用（DashScope 等网关的思考模式对非流式调用强制 400，必须流式）")


class FrameworkConfig(BaseModel):
    """框架总配置。"""

    agents: list[AgentConfig] = Field(default_factory=list, description="Agent 配置列表")
    topology: TopologyConfig = Field(default_factory=TopologyConfig, description="拓扑策略配置")
    decision: DecisionConfig = Field(default_factory=DecisionConfig, description="决策策略配置")
    routing: RoutingConfig = Field(default_factory=RoutingConfig, description="路由策略配置")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM 配置")
    max_rounds: int = Field(default=10, ge=1, description="全局最大执行轮次")
    verbose: bool = Field(default=True, description="是否输出详细日志")
    log_dir: Optional[str] = Field(default=None, description="日志输出目录")
    description: Optional[str] = Field(default=None, description="实验描述")
    memory: Optional[dict] = Field(default=None, description="记忆更新策略配置，如 {'strategy': 'ledger'} 或 {'strategy': 'default'}")
    initial_goal: Optional[str] = Field(
        default=None,
        description="首轮初始目标（覆盖默认的 'Analyze the problem and propose an initial approach.'）",
    )
    srac: Optional[dict] = Field(
        default=None,
        description="Speech-act Role Attention Completion (SRAC) 配置，如 {'enabled': true, 'mode': 'anomaly_only', ...}",
    )

    def to_dict(self) -> dict:
        """导出为字典。"""
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict) -> "FrameworkConfig":
        """从字典构建。"""
        return cls.model_validate(data)

    def to_json_schema(self) -> dict:
        """生成 JSON Schema，供 GUI 自动生成表单。"""
        return self.model_json_schema()
