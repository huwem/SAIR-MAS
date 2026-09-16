# MAS_Topo 配置文件说明

## 目录结构

```
configs/
├── README.md                          # 本文件
├── _template_experiment.yaml          # 实验级配置完整模板（含所有字段注释）
├── fixed_chain.yaml                   # 纯框架配置示例（固定链式拓扑，适合直接 run）
└── experiments/                       # 现成实验配置（方法 × 任务类型）
    ├── vanilla_code_local.yaml        # 单智能体基线（代码任务）
    ├── vanilla_reasoning_local.yaml   # 单智能体基线（推理任务）
    ├── autogen_code_local.yaml        # AutoGen 风格 GroupChat（代码任务）
    ├── autogen_reasoning_local.yaml   # AutoGen 风格 GroupChat（推理任务）
    ├── dytopo_code_local.yaml         # DyTopo 动态拓扑（代码任务）
    ├── dytopo_reasoning_local.yaml    # DyTopo 动态拓扑（推理任务）
    ├── p2p_free_code_local.yaml       # P2P-Free 自由通信（代码任务）
    ├── p2p_free_reasoning_local.yaml  # P2P-Free 自由通信（推理任务）
    ├── sair_code_local.yaml           # SAIR 完整框架（代码任务）
    ├── sair_reasoning_local.yaml      # SAIR 完整框架（推理任务）
    └── ablations/                     # 消融配置
        ├── component/                 #   组件消融（如去 SRAC）
        └── topo/                      #   拓扑受限消融（如 chain 拓扑）
```

## 两种配置模式

### 1. 实验级配置（推荐）

用于在数据集上批量运行实验，封装了数据集、评估指标和框架参数。

文件名约定：`<method>_<code|reasoning>_local.yaml`（code 任务对应 HumanEval/LiveCodeBench，reasoning 任务对应 MATH/AIME）。

启动方式：

```bash
python -m mas_topo configs/experiments/vanilla_code_local.yaml -l 5
python -m mas_topo configs/experiments/sair_reasoning_local.yaml -l 10 -b 3
```

顶层字段：

- `name` / `description` — 实验元信息（可选）
- `dataset` — 数据集配置（loader, path, limit, offset 等）
- `evaluation` — 评估配置（metric, metric_params）
- `output` — 输出配置（dir, formats, trace）
- `framework` — 框架配置（agents, topology, routing, decision, llm 等）

### 2. 纯框架配置（向后兼容）

仅定义多智能体框架参数（即实验级配置中的 `framework` 段单独保存为一个 YAML 文件），适合直接运行单个任务或编程式调用。

启动方式：

```bash
python -m mas_topo run configs/fixed_chain.yaml "Write a Python function..."
```

顶层字段与实验级配置中的 `framework` 部分相同，没有 `dataset` / `evaluation` / `output` 层。

## 核心字段详解

### dataset（实验级配置必填）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `loader` | string | `"humaneval"` | 数据集加载器注册表名。可用：`humaneval`, `math`, `aime`, `livecodebench` |
| `path` | string | `""` | 数据集文件路径（相对项目根目录或绝对路径） |
| `limit` | int / null | `null` | 最多运行的样本数，`null` 表示全部 |
| `offset` | int | `0` | 起始偏移量 |
| `random_offset` | bool | `false` | 是否随机选择起始偏移 |
| `seed` | int / null | `null` | 随机种子 |

### evaluation（实验级配置可选）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `metric` | string | `"exact_match"` | 评估器注册表名。可用：`exact_match`, `code_execution`, `code_execution_stdin`, `llm_judge` |
| `metric_params` | dict | `{}` | 评估器额外参数 |

`metric_params` 示例：

- `llm_judge`：`{model: deepseek-v4-flash, api_key: ${OPENAI_API_KEY}, base_url: ${OPENAI_BASE_URL}, temperature: 0.0, max_tokens: 256}` — 大模型评判，适合 MATH/AIME 等复杂推理任务
- `exact_match`：`{extract_number: true}` — 从文本中提取数字后匹配（仅推荐用于 GSM8K 等纯数字答案任务）
- `code_execution`：`{extract_code: true}` — 从 markdown 代码块中提取 Python 代码执行

### output（实验级配置可选）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dir` | string / null | `null` | 输出目录，`null` 则自动生成带时间戳的目录 |
| `formats` | list | `["json", "csv"]` | 结果导出格式 |
| `trace` | bool | `true` | 是否启用过程追踪（每轮 Agent I/O、拓扑、路由决策） |

### framework 核心字段

#### llm

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `provider` | string | `"openai_compat"` | LLM 适配器注册表名 |
| `model` | string | `"gpt-4o"` | 模型名称 |
| `api_key` | string / null | `null` | API 密钥（推荐用 `${OPENAI_API_KEY}` 环境变量） |
| `base_url` | string / null | `null` | API 基础 URL（推荐用 `${OPENAI_BASE_URL}` 环境变量） |
| `temperature` | float | `0.2` | 采样温度 (0.0~2.0) |
| `max_tokens` | int | `1000` | 最大生成 token 数 |
| `json_mode` | bool | `false` | 是否启用 OpenAI JSON mode |

#### agents（列表）

每个 Agent 的字段：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `type` | string | 是 | Agent 类型注册表名（目前只有 `LLMAgent`） |
| `name` | string | 是 | Agent 唯一名称 |
| `role` | string | 否 | 角色描述 |
| `system_prompt` | string | 否 | 系统提示词（覆盖默认角色提示） |
| `output_format` | string | 否 | 输出格式：`"text"` 或 `"json"` |
| `output_schema` | dict | 否 | JSON 输出字段定义（LLM 会收到 schema 说明） |
| `required_output_fields` | list | 否 | 必填字段列表，用于 Agent 输出验证 |
| `tools` | list | 否 | 工具名称列表（对应 tool_registry） |
| `memory_limit` | int | 否 | 记忆历史截断长度，默认 3000 |

`required_output_fields` 可用值：

- `public_content` — 必须非空字符串
- `q_vector` — 映射到 query_descriptor
- `k_vector` — 映射到 key_descriptor
- `messages` — FT 模式下 Worker 间通信的消息列表
- `answer` — 最终答案
- `is_complete` — Manager 的完成标志
- `next_goal` — Manager 的下轮目标

#### topology

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | string | `"semantic"` | 拓扑策略名 |
| `pattern` | string / null | `null` | fixed 策略专用：star, chain, ring, full_connected, tree, mesh, layered, random |
| `threshold` | float | `0.3` | semantic/cosine 策略：相似度阈值 |
| `max_in_degree` | int | `3` | semantic/cosine 策略：最大入度 |
| `embedding_model` | string | `"all-MiniLM-L6-v2"` | semantic/cosine 策略：嵌入模型路径 |
| `manager_name` | string / null | `null` | Manager 名称，启用后拓扑仅计算 Worker 间连接 |

可用拓扑策略：

- `fixed` — 预定义固定拓扑（通过 `pattern` 指定形状）
- `semantic` — 语义匹配拓扑（基于 q_vector/k_vector 的嵌入相似度）
- `cosine` — 余弦相似度拓扑
- `manual` — 手动指定邻接矩阵（通过 `adjacency` 字段）

#### routing

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | string | `"semantic"` | 路由策略名 |
| `memory_truncate` | int | `500` | 消息内容截断长度 |
| `manager_name` | string / null | `null` | Manager 名称，启用后分离 public/private 信道 |

可用路由策略：

- `semantic` — 语义路由（基于 q_vector/k_vector 匹配，优先 private_content）
- `free_talk` — 显式 to 路由（Worker 间通过 messages[].to 投递）
- `graph` — 图邻接路由（沿拓扑边传递 public_content）
- `broadcast` — 广播路由（全对全通信）

#### decision

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `strategy` | string | `"manager"` | 决策策略名 |
| `manager_name` | string | `"Manager"` | manager 策略专用 Manager Agent 名称 |
| `system_prompt` | string / null | `null` | 自定义决策提示词 |
| `max_rounds` | int | `10` | 最大执行轮次 |
| `phase5_prompt_template` | string / null | `null` | 覆盖默认 Phase 5 prompt 模板（可选） |

可用决策策略：

- `manager` — Manager Agent 二次调用决定 is_complete/next_goal
- `voting` — 多数投票决策
- `direct` — 直接取最后一个 Agent 输出
- `llm_judge` — 独立 LLM 裁决

#### 全局框架字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_rounds` | int | `10` | 全局最大执行轮次 |
| `verbose` | bool | `true` | 是否输出详细日志 |
| `log_dir` | string / null | `null` | 日志输出目录 |
| `description` | string / null | `null` | 实验描述 |
| `initial_goal` | string / null | `null` | 首轮初始目标（覆盖默认的 "Analyze the problem and propose an initial approach."） |

## 环境变量

配置文件支持 `${ENV_VAR}` 语法，推荐通过环境变量传入敏感信息：

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.deepseek.com"
export OPENAI_MODEL="deepseek-v4-flash"
```

## 现成配置文件说明

| 文件 | 模式 | 用途 |
|------|------|------|
| `experiments/vanilla_code_local.yaml` / `experiments/vanilla_reasoning_local.yaml` | 实验级 | 单智能体基线（代码 / 推理） |
| `experiments/autogen_code_local.yaml` / `experiments/autogen_reasoning_local.yaml` | 实验级 | AutoGen 风格 GroupChat + Manager |
| `experiments/dytopo_code_local.yaml` / `experiments/dytopo_reasoning_local.yaml` | 实验级 | DyTopo 语义动态拓扑 |
| `experiments/p2p_free_code_local.yaml` / `experiments/p2p_free_reasoning_local.yaml` | 实验级 | P2P-Free 显式自由通信 |
| `experiments/sair_code_local.yaml` / `experiments/sair_reasoning_local.yaml` | 实验级 | SAIR 完整框架 |
| `experiments/ablations/` | 实验级 | 组件消融（component/）与拓扑受限消融（topo/） |
| `fixed_chain.yaml` | 纯框架 | 固定链式拓扑示例（适合直接 run 单个任务） |
| `_template_experiment.yaml` | 模板 | 完整字段注释，编写新配置的参考 |

## 编写新配置的最小示例

```yaml
name: "quick_test"
dataset:
  loader: humaneval
  path: datasets/humaneval/humaneval-py.jsonl
  limit: 5
evaluation:
  metric: code_execution
framework:
  llm:
    provider: openai_compat
    model: ${OPENAI_MODEL}
    api_key: ${OPENAI_API_KEY}
    base_url: ${OPENAI_BASE_URL}
  agents:
    - type: LLMAgent
      name: Manager
      role: Coordinator
    - type: LLMAgent
      name: Worker
      role: Solver
  topology:
    strategy: fixed
    pattern: star
  routing:
    strategy: broadcast
  decision:
    strategy: direct
```
