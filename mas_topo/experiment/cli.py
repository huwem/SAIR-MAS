"""通用实验 CLI 逻辑。

支持通过单个配置文件运行任意数据集实验：
    python -m mas_topo experiment configs/my_experiment.yaml

实验配置（ExperimentConfig）封装了数据集、评估指标和框架参数，
实现"零代码"运行实验。
"""

import argparse
import logging
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from mas_topo.config.experiment_schema import ExperimentConfig
from mas_topo.config.schema import FrameworkConfig
from mas_topo.config.loader import load_config, _substitute_env_vars
from mas_topo._registries import dataset_registry, evaluator_registry
from mas_topo.datasets.base import BaseDatasetLoader
from mas_topo.evaluation.base import Evaluator
from mas_topo.experiment.dataset_runner import DatasetExperimentRunner


# 内置数据集预设：通过 --dataset 参数一键切换数据集，无需复制配置文件。
DATASET_PRESETS: dict[str, dict] = {
    "aime": {
        "loader": "aime",
        "path": "datasets/aime/aime_*.jsonl",
        "metric": "llm_judge",
        "metric_params": {
            "model": "deepseek-v4-flash",
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "${OPENAI_BASE_URL}",
            "temperature": 0.0,
            "max_tokens": 256,
        },
    },
    "math": {
        "loader": "math",
        "path": "datasets/math/test.jsonl",
        "metric": "llm_judge",
        "metric_params": {
            "model": "deepseek-v4-flash",
            "api_key": "${OPENAI_API_KEY}",
            "base_url": "${OPENAI_BASE_URL}",
            "temperature": 0.0,
            "max_tokens": 256,
        },
    },
    "humaneval": {
        "loader": "humaneval",
        "path": "datasets/humaneval/humaneval-py.jsonl",
        "metric": "code_execution",
        "metric_params": {"extract_code": True},
    },
    "livecodebench": {
        "loader": "livecodebench",
        "path": "datasets/livecodebench/test6_atcoder.jsonl",
        "metric": "code_execution_stdin",
        "metric_params": {"extract_code": True},
    },
}


def setup_logging(level_name: str, extra_noisy: Optional[list[str]] = None) -> None:
    """配置日志级别，降低第三方库噪音。"""
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }
    level = level_map.get(level_name.lower(), logging.WARNING)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    noisy = [
        "httpx",
        "httpcore",
        "openai",
        "urllib3",
        "sentence_transformers",
    ]
    if extra_noisy:
        noisy.extend(extra_noisy)
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)


def build_experiment_parser() -> argparse.ArgumentParser:
    """构建实验子命令的参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Run multi-agent experiments via configuration file",
        prog="python -m mas_topo experiment",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to experiment YAML/JSON config file",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Override dataset limit from config",
    )
    parser.add_argument(
        "--offset", "-o",
        type=int,
        default=None,
        help="Override dataset offset from config",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Override LLM model name",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override LLM API base URL",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Override LLM API key (also reads OPENAI_API_KEY env if not set)",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        choices=list(DATASET_PRESETS.keys()),
        help="Override dataset by preset name (also sets default path and metric)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override dataset file path (keeps the metric from --dataset or config)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Override batch size (concurrent samples)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override LLM max_tokens (useful to cap thinking-mode cost/latency)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=None,
        help="Override total LLM request timeout in seconds (0 disables it)",
    )
    parser.add_argument(
        "--sample-timeout",
        type=float,
        default=None,
        help="Override per-sample wall-clock timeout in seconds (0 disables it)",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Disable model thinking/reasoning phase (sends enable_thinking=false; "
             "required by DashScope-style gateways for non-streaming calls)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming calls (required for thinking mode on DashScope-style "
             "gateways, which reject non-streaming enable_thinking=true)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="warning",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Local SentenceTransformer model path",
    )
    parser.add_argument(
        "--cache-folder",
        type=str,
        default=None,
        help="SentenceTransformer cache directory",
    )
    return parser


def run_experiment_from_config(
    config_path: str,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_path: Optional[str] = None,
    batch_size: Optional[int] = None,
    max_tokens: Optional[int] = None,
    request_timeout: Optional[float] = None,
    sample_timeout: Optional[float] = None,
    disable_thinking: bool = False,
    stream: bool = False,
    output_dir: Optional[str] = None,
    log_level: str = "warning",
    local_model: Optional[str] = None,
    cache_folder: Optional[str] = None,
) -> None:
    """从配置文件运行实验。

    完整流程：加载配置 → 获取 loader/evaluator → 设置日志 → 运行实验 → 输出结果。
    """
    setup_logging(log_level)

    if local_model:
        os.environ["ST_MODEL_PATH"] = local_model
        os.environ["ST_LOCAL_ONLY"] = "true"
    if cache_folder:
        os.environ["ST_CACHE_FOLDER"] = cache_folder

    # 1. 加载配置
    print(f"Loading experiment config from: {config_path}")
    loaded = load_config(config_path)

    if isinstance(loaded, ExperimentConfig):
        exp_config = loaded
        fw_config = exp_config.framework
    else:
        # 向后兼容：纯框架配置时，尝试构造最小实验配置
        fw_config = loaded
        exp_config = ExperimentConfig(framework=fw_config)
        print("[Warning] Loaded a framework-only config. Using default dataset/evaluation settings.")

    # 2. 命令行参数覆盖配置
    if model:
        fw_config.llm.model = model
    if base_url:
        fw_config.llm.base_url = base_url
    if api_key:
        fw_config.llm.api_key = api_key
    if request_timeout is not None:
        fw_config.llm.request_timeout = request_timeout
    if max_tokens is not None:
        fw_config.llm.max_tokens = max_tokens
    if disable_thinking:
        fw_config.llm.disable_thinking = True
    if stream:
        fw_config.llm.stream = True
    if not fw_config.llm.api_key and os.environ.get("OPENAI_API_KEY"):
        fw_config.llm.api_key = os.environ.get("OPENAI_API_KEY")
    if not fw_config.llm.base_url and os.environ.get("OPENAI_BASE_URL"):
        fw_config.llm.base_url = os.environ.get("OPENAI_BASE_URL")

    ds_cfg = exp_config.dataset
    eval_cfg = exp_config.evaluation

    # 命令行覆盖数据集与评估指标
    if dataset:
        preset = DATASET_PRESETS[dataset]
        ds_cfg.loader = preset["loader"]
        ds_cfg.path = preset["path"]
        eval_cfg.metric = preset["metric"]
        eval_cfg.metric_params = _substitute_env_vars(preset["metric_params"])
    if dataset_path:
        ds_cfg.path = dataset_path

    # 评估器（尤其是 llm_judge）的 API 参数应继承主 LLM 配置，
    # 避免命令行 --api-key/--base-url 只覆盖了 framework.llm 而评估器仍用旧值。
    if eval_cfg.metric == "llm_judge":
        eval_params = dict(eval_cfg.metric_params)
        if model and "model" not in eval_params:
            eval_params["model"] = model
        if base_url and not eval_params.get("base_url"):
            eval_params["base_url"] = base_url
        if api_key and not eval_params.get("api_key"):
            eval_params["api_key"] = api_key
        if not eval_params.get("api_key") and fw_config.llm.api_key:
            eval_params["api_key"] = fw_config.llm.api_key
        if not eval_params.get("base_url") and fw_config.llm.base_url:
            eval_params["base_url"] = fw_config.llm.base_url
        eval_cfg.metric_params = eval_params

    out_cfg = exp_config.output

    actual_limit = limit if limit is not None else ds_cfg.limit
    actual_offset = offset if offset is not None else ds_cfg.offset
    actual_batch = batch_size if batch_size is not None else 1
    actual_sample_timeout = (
        sample_timeout if sample_timeout is not None else ds_cfg.sample_timeout
    )

    # 3. 从注册表动态获取 loader 和 evaluator
    if not dataset_registry.has(ds_cfg.loader):
        available = dataset_registry.list()
        raise KeyError(
            f"Dataset loader '{ds_cfg.loader}' not found. "
            f"Available: {available}. Did you register it?"
        )
    loader = dataset_registry.get(ds_cfg.loader, data_path=ds_cfg.path)

    if not evaluator_registry.has(eval_cfg.metric):
        available = evaluator_registry.list()
        raise KeyError(
            f"Evaluator '{eval_cfg.metric}' not found. "
            f"Available: {available}. Did you register it?"
        )
    evaluator = evaluator_registry.get(eval_cfg.metric, **eval_cfg.metric_params)

    total_available = loader.total_count()
    print(f"\nDataset: {ds_cfg.loader} ({ds_cfg.path})")
    print(f"  Total samples available: {total_available}")

    if ds_cfg.random_offset:
        if ds_cfg.seed is not None:
            random.seed(ds_cfg.seed)
        actual_limit = actual_limit if actual_limit is not None else total_available
        max_off = max(0, total_available - actual_limit)
        actual_offset = random.randint(0, max_off)
        print(f"  Random offset: {actual_offset} (seed={ds_cfg.seed})")

    print(f"\nFramework:")
    print(f"  Agents: {[a.name for a in fw_config.agents]}")
    print(f"  Topology: {fw_config.topology.strategy}")
    print(f"  Routing: {fw_config.routing.strategy}")
    print(f"  Decision: {fw_config.decision.strategy}")
    print(f"  Model: {fw_config.llm.model}")
    print(f"  Evaluator: {eval_cfg.metric}")

    # 4. 确定输出目录
    if output_dir:
        actual_output_dir = output_dir
    elif out_cfg.dir:
        actual_output_dir = out_cfg.dir
    else:
        name_tag = exp_config.name or Path(ds_cfg.path).stem
        # 数据集标签：humaneval/livecodebench 共用 code 配置、math/aime 共用
        # reasoning 配置，仅按配置名+时间戳命名会在并发启动时目录同名、
        # results.json 互相覆写，因此必须带上数据集段。
        if dataset_path:
            ds_tag = Path(dataset_path).stem
        elif dataset:
            ds_tag = dataset
        else:
            ds_tag = Path(ds_cfg.path).stem
        ds_tag = re.sub(r"[^A-Za-z0-9_\-]+", "_", ds_tag).strip("_")
        model_tag = fw_config.llm.model.split("/")[-1]
        topo_tag = fw_config.topology.strategy
        routing_tag = fw_config.routing.strategy
        dec_tag = fw_config.decision.strategy
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        actual_output_dir = f"experiment_results/{name_tag}_{ds_tag}_{topo_tag}_{routing_tag}_{dec_tag}_{model_tag}_{ts}"

    fw_config.log_dir = actual_output_dir
    os.makedirs(actual_output_dir, exist_ok=True)

    # 5. 设置文件日志
    log_path = os.path.join(actual_output_dir, "experiment.log")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
    ))
    mas_logger = logging.getLogger("mas_topo")
    mas_logger.addHandler(file_handler)
    mas_logger.setLevel(logging.INFO)
    mas_logger.propagate = False

    # 6. 运行实验
    runner = DatasetExperimentRunner(
        config=fw_config,
        evaluator=evaluator,
        verbose=True,
        output_dir=actual_output_dir,
        progress_bar=True,
    )
    summary = runner.run_on_dataset(
        loader,
        limit=actual_limit,
        offset=actual_offset,
        batch_size=actual_batch,
        sample_timeout=actual_sample_timeout,
    )

    # 7. 输出汇总
    runner.print_summary(summary)
    print(f"\nDone. Results exported to: {actual_output_dir}")
    print(f"  Log: {log_path}")


def experiment_command(args: Optional[list[str]] = None) -> None:
    """实验子命令入口。"""
    parser = build_experiment_parser()
    parsed = parser.parse_args(args)
    run_experiment_from_config(
        config_path=parsed.config,
        limit=parsed.limit,
        offset=parsed.offset,
        model=parsed.model,
        base_url=parsed.base_url,
        api_key=parsed.api_key,
        dataset=parsed.dataset,
        dataset_path=parsed.dataset_path,
        batch_size=parsed.batch_size,
        max_tokens=parsed.max_tokens,
        request_timeout=parsed.request_timeout,
        sample_timeout=parsed.sample_timeout,
        disable_thinking=parsed.disable_thinking,
        stream=parsed.stream,
        output_dir=parsed.output_dir,
        log_level=parsed.log_level,
        local_model=parsed.local_model,
        cache_folder=parsed.cache_folder,
    )


def run_task_command() -> None:
    """直接运行单个任务（向后兼容）。"""
    parser = argparse.ArgumentParser(
        description="Run a single task with a framework config",
        prog="python -m mas_topo run",
    )
    parser.add_argument("config", type=str, help="Framework config file path")
    parser.add_argument("task", type=str, help="Task text")
    parser.add_argument("--log-level", type=str, default="warning",
                        choices=["debug", "info", "warning", "error", "critical"])
    parsed = parser.parse_args()

    setup_logging(parsed.log_level)
    fw_config = load_config(parsed.config)
    if isinstance(fw_config, ExperimentConfig):
        fw_config = fw_config.framework

    if not fw_config.llm.api_key and os.environ.get("OPENAI_API_KEY"):
        fw_config.llm.api_key = os.environ.get("OPENAI_API_KEY")
    if not fw_config.llm.base_url and os.environ.get("OPENAI_BASE_URL"):
        fw_config.llm.base_url = os.environ.get("OPENAI_BASE_URL")

    from mas_topo.graph import Graph
    graph = Graph.from_config(fw_config)
    result = graph.run(parsed.task)
    print(f"\nCompleted: {result.is_completed}")
    print(f"Rounds: {len(result.rounds)}")
    print(f"Final answer: {result.final_answer}")
