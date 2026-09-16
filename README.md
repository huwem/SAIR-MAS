# SAIR-MAS

A configuration-driven multi-agent topology experimentation framework (Python package: `mas_topo`). Define agent teams, topology structures, routing strategies, and decision mechanisms through YAML configuration files, and quickly run experiments on datasets such as HumanEval, MATH, and AIME.

---

## English

### Quick Start

Run an example experiment in 5 minutes:

```bash
# 1. Create and activate a conda environment
conda create -n mas_topo python=3.11 -y
conda activate mas_topo

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure the LLM API (see "Configure the LLM API" below)

# 4. Run an example experiment (HumanEval, single-agent baseline)
python -m mas_topo configs/experiments/vanilla_code_local.yaml -l 3
```

> Running experiments calls the LLM API and consumes tokens. Make sure your API key is valid and has sufficient balance.

### Environment

**Python 3.11 or higher** is recommended.

```bash
conda create -n mas_topo python=3.11 -y
conda activate mas_topo
```

If conda is not installed yet, see the [Miniconda installation guide](https://docs.conda.io/en/latest/miniconda.html).

### Install Dependencies

Core dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

For **semantic topology** (`semantic` / `cosine`) or **neural routing**, additionally install:

```bash
pip install sentence-transformers torch
```

For the **visualization tools**, additionally install:

```bash
pip install matplotlib networkx pandas flask Pillow
```

### Configure the LLM API

The LLM connection is configured in the `llm` section of each YAML config. The shipped configs use `api_key: sk-placeholder` — replace it with your own key, or use the `${ENV_VAR}` syntax to read from environment variables:

```yaml
llm:
  provider: openai_compat
  model: ${OPENAI_MODEL}        # e.g. deepseek-chat
  api_key: ${OPENAI_API_KEY}    # or paste your key directly (not recommended)
  base_url: ${OPENAI_BASE_URL}  # any OpenAI-compatible endpoint
```

Then export the variables before running:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com"   # or any OpenAI-compatible service
export OPENAI_MODEL="deepseek-chat"                 # or another model name
```

Alternatively, copy `.env.example` to `.env`, fill in your keys, and `source .env` before running. This convention (per-model environment variables) is used by `scripts/run_experiment_matrix.py`. Never commit real keys to version control.

### Running Experiments

#### Mode 1: Experiment-level configs (recommended)

Experiment-level configs define the dataset, evaluation metric, and framework parameters together — suitable for batch runs on a dataset:

```bash
python -m mas_topo configs/experiments/sair_code_local.yaml -l 5 -b 2
```

Common arguments:

- `-l, --limit N`: run only the first N samples
- `-b, --batch-size B`: run B samples concurrently
- `-m, --model MODEL`: override the model name in the config
- `--log-level LEVEL`: debug / info / warning / error / critical

Ready-made configs live in `configs/experiments/` (one per method × task type), with ablation configs under `configs/experiments/ablations/`. To run a full matrix of methods × datasets × models, see `scripts/run_experiment_matrix.py`.

#### Mode 2: Pure framework configs (single task)

A pure framework config defines only the multi-agent framework (the `framework` section of an experiment-level config saved as its own YAML file), suitable for running a single task directly. A ready-made example is shipped as `configs/fixed_chain.yaml`:

```bash
python -m mas_topo run configs/fixed_chain.yaml "Write a Python function that returns the sum of two numbers."
```

### Verify the Installation

Run the unit tests:

```bash
python -m pytest -q
```

Or simply check that the package imports:

```bash
python -c "import mas_topo; print('mas_topo OK')"
```

### Datasets

This release includes the **AIME**, **HumanEval**, and **MATH** datasets under `datasets/`. **LiveCodeBench** is not included (approx. 195 MB). To run LiveCodeBench experiments, download it from the official sources and place it under `datasets/livecodebench/`:

- Official repository: https://github.com/LiveCodeBench/LiveCodeBench
- Hugging Face: https://huggingface.co/datasets/livecodebench/livecodebench

### Project Structure

```
.
├── configs/            # Experiment & framework configs
│   ├── experiments/    #   Ready-made experiment configs (incl. ablations/)
│   ├── fixed_chain.yaml           # Pure-framework config example (single task)
│   └── _template_experiment.yaml  # Fully-commented config template
├── mas_topo/           # Core framework package (incl. tests/)
├── datasets/           # Bundled datasets: aime/ humaneval/ math/
├── scripts/            # Helper scripts (run experiment matrix, aggregate results)
├── visualization/      # Result visualization tools
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── LICENSE             # MIT License
└── README.md           # This file
```

### Reproducing the Experimental Results

Raw experiment outputs (`experiment_results/`) are not included in this release. To reproduce the results, re-run the experiments with the configs under `configs/experiments/` (and `configs/experiments/ablations/` for the ablation studies), e.g.:

```bash
python -m mas_topo configs/experiments/sair_code_local.yaml          # full dataset
python scripts/run_experiment_matrix.py --method sair --dataset all --model all
```

Results are written to `experiment_results/<run_name>_<timestamp>/`. Afterwards, `scripts/aggregate_experiment_results.py` aggregates all runs into a single summary CSV, and the tools under `visualization/` can inspect rounds, routing, and metrics.

### Next Steps

- Full configuration reference: [configs/README.md](configs/README.md)
- Put additional datasets under `datasets/` to run other benchmarks.

---

## 中文

MAS_Topo 是一个配置驱动的多智能体拓扑实验框架。通过 YAML 配置文件定义 Agent 团队、拓扑结构、路由策略与决策机制，可快速在 HumanEval、MATH、AIME 等数据集上运行实验。

### 快速开始

5 分钟内跑通一个示例实验：

```bash
# 1. 创建并激活 conda 环境
conda create -n mas_topo python=3.11 -y
conda activate mas_topo

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 LLM API（见下文"配置 LLM API"）

# 4. 运行一个示例实验（以 HumanEval 单智能体基线为例）
python -m mas_topo configs/experiments/vanilla_code_local.yaml -l 3
```

> 运行实验会调用 LLM API 并消耗 token，请确保 API 密钥可用且余额充足。

### 环境创建

推荐使用 **Python 3.11 或更高版本**。如果还没有安装 conda，可参考 [Miniconda 安装指南](https://docs.conda.io/en/latest/miniconda.html)。

### 安装依赖

项目依赖列在 `requirements.txt` 中，安装核心依赖即可运行大部分示例。若需要使用**语义拓扑**（`semantic` / `cosine`）或**神经网络路由**，还需安装 `sentence-transformers torch`；若需要使用**可视化工具**，还需安装 `matplotlib networkx pandas flask Pillow`（命令同上英文节）。

### 配置 LLM API

LLM 连接在各 YAML 配置的 `llm` 段中设置。随附配置中的 `api_key: sk-placeholder` 为占位符，请替换为你自己的密钥；配置文件也支持 `${ENV_VAR}` 语法读取环境变量：

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.deepseek.com"   # 或其他 OpenAI 兼容服务
export OPENAI_MODEL="deepseek-chat"                 # 或其他模型名称
```

也可以复制 `.env.example` 为 `.env`，填入真实密钥后 `source .env`（`scripts/run_experiment_matrix.py` 使用这种按模型命名的环境变量约定）。请勿将真实密钥提交到版本控制。

### 运行实验

**方式一：实验级配置（推荐）**——同时定义数据集、评估指标和框架参数，适合批量跑数据集：

```bash
python -m mas_topo configs/experiments/sair_code_local.yaml -l 5 -b 2
```

常用参数：`-l, --limit N` 只跑前 N 条样本；`-b, --batch-size B` 并发运行 B 个样本；`-m, --model MODEL` 覆盖配置中的模型名称；`--log-level LEVEL` 日志级别。现成配置位于 `configs/experiments/`，消融配置位于 `configs/experiments/ablations/`；批量矩阵运行见 `scripts/run_experiment_matrix.py`。

**方式二：纯框架配置（运行单个任务）**——只定义多智能体框架（即实验级配置中的 `framework` 段单独保存为 YAML），适合直接运行单个任务。随附示例为 `configs/fixed_chain.yaml`：

```bash
python -m mas_topo run configs/fixed_chain.yaml "Write a Python function that returns the sum of two numbers."
```

### 验证安装

```bash
python -m pytest -q
python -c "import mas_topo; print('mas_topo OK')"
```

### 数据集说明

本发布包中 `datasets/` 目录包含 **AIME**、**HumanEval**、**MATH** 数据集，**未包含 LiveCodeBench**（体积约 195 MB）。如需运行 LiveCodeBench 相关实验，请从官方渠道获取后放到 `datasets/livecodebench/` 目录下：

- 官方仓库：https://github.com/LiveCodeBench/LiveCodeBench
- Hugging Face：https://huggingface.co/datasets/livecodebench/livecodebench

### 项目结构

```
.
├── configs/            # 实验与框架配置文件
│   ├── experiments/    #   现成实验配置（含 ablations/ 消融配置）
│   ├── fixed_chain.yaml           # 纯框架配置示例（运行单个任务）
│   └── _template_experiment.yaml  # 含全部字段注释的配置模板
├── mas_topo/           # 核心框架代码（含 tests/ 单元测试）
├── datasets/           # 随附数据集：aime/ humaneval/ math/
├── scripts/            # 辅助脚本（实验矩阵运行、结果汇总）
├── visualization/      # 实验结果可视化工具
├── requirements.txt    # Python 依赖
├── .env.example        # 环境变量模板
├── LICENSE             # MIT 许可证
└── README.md           # 本文件
```

### 复现实验结果

本发布包不包含原始实验输出（`experiment_results/`）。如需复现论文中的结果，请使用 `configs/experiments/`（及消融实验的 `configs/experiments/ablations/`）下的配置重新运行实验：

```bash
python -m mas_topo configs/experiments/sair_code_local.yaml          # 跑完整数据集
python scripts/run_experiment_matrix.py --method sair --dataset all --model all
```

结果会写入 `experiment_results/<实验名>_<时间戳>/`。之后可用 `scripts/aggregate_experiment_results.py` 将所有实验汇总为一张 CSV 表，也可以用 `visualization/` 下的工具查看轮次、路由与指标细节。

### 下一步

- 查看完整配置说明：[configs/README.md](configs/README.md)
- 将其他数据集放到 `datasets/` 目录下即可运行更多基准。
