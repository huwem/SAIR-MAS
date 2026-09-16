#!/usr/bin/env python3
"""
实验矩阵启动脚本：支持 4 个主方法（+2 个消融臂，需显式指定）× 4 种数据集 × 3 种模型的任意组合。

约定：
- 方法（method）: vanilla / dytopo / p2p_free / sair（主矩阵方法，`all` 只展开到这 4 个）
  消融臂 sair_no_srac / sair_no_health 需显式指定，不会随 `all` 展开
  对应 configs/experiments/ 下的 {method}_{code|reasoning}_local.yaml
- 数据集（dataset）: humaneval / livecodebench / math / aime（排除 GPQA）
  humaneval、livecodebench 使用 code 配置；math、aime 使用 reasoning 配置
- 模型（model）: deepseek / qwen3 / gptoss120b
  所有模型的 API key 和 base URL 均从项目根目录的 .env 读取，变量名模型专属：
  - deepseek:     DEEPSEEK_API_KEY       / DEEPSEEK_BASE_URL
  - qwen3:        QWEN3_API_KEY          / QWEN3_BASE_URL
  - gptoss120b:   GPT_OSS_120B_API_KEY   / GPT_OSS_120B_BASE_URL

用法：
    # 先加载 .env
    source .env

    # 跑单个组合
    python scripts/run_experiment_matrix.py --method p2p_free --dataset humaneval --model deepseek -l 5

    # 跑某个方法在所有数据集、所有模型上的完整矩阵
    python scripts/run_experiment_matrix.py --method p2p_free --dataset all --model all

    # 跑所有方法在单个数据集、单个模型上的矩阵
    python scripts/run_experiment_matrix.py --method all --dataset math --model qwen3
    nohup python scripts/run_experiment_matrix.py --method sair --dataset math --model qwen3 > sair_math_qwen3_experiment.log 2>&1 &

    # 仅打印命令，不实际执行
    python scripts/run_experiment_matrix.py --method dytopo --dataset humaneval --model gptoss120b --dry-run

    # 并发跑矩阵（4 个实验并行）
    python scripts/run_experiment_matrix.py --method all --dataset math --model qwen3 -j 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
ABLATION_COMPONENT_DIR = CONFIG_DIR / "ablations" / "component"

# 方法 -> 配置文件前缀（4 个主矩阵方法 + 2 个消融臂，消融需显式指定）
METHODS = {
    "vanilla": "vanilla",
    "dytopo": "dytopo",
    "sair": "sair",
    "p2p_free": "p2p_free",
    "autogen": "autogen",
    "sair_no_srac": "sair_no_srac",
    "sair_no_health": "sair_no_health",
    # 剂量上限臂：SRAC 常开，但每题最多交付 N 条补全
    "sair_dose1": "sair_dose1",
    "sair_dose2": "sair_dose2",
    "sair_dose_r1": "sair_dose_r1",
    "sair_dose_t70": "sair_dose_t70",
    "sair_dose_r1t70": "sair_dose_r1t70",
}

# 主矩阵方法：`--method all` 只展开到这四个，消融臂(sair_no_srac/sair_no_health)需显式指定
MAIN_METHODS = ["vanilla", "dytopo", "sair", "p2p_free"]

# 4 种数据集：code 类用 code 配置，reasoning 类用 reasoning 配置
DATASETS = ["humaneval", "livecodebench", "math", "aime"]
CODE_DATASETS = {"humaneval", "livecodebench"}
REASONING_DATASETS = {"math", "aime"}

# 3 种模型：所有 endpoint / key 均从 .env 读取，避免与系统环境变量混淆
MODELS = {
    "deepseek": {
        "name": "deepseek-v4-flash",
        "base_url": None,            # 从 DEEPSEEK_BASE_URL 读取
        "api_key": None,             # 从 DEEPSEEK_API_KEY 读取
        "local": False,
        "health_url": None,
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_BASE_URL",
    },
    "qwen3": {
        "name": "qwen3-32b",
        "base_url": None,            # 从 QWEN3_BASE_URL 读取
        "api_key": None,             # 从 QWEN3_API_KEY 读取
        "local": False,
        "health_url": None,
        "api_key_env": "QWEN3_API_KEY",
        "base_url_env": "QWEN3_BASE_URL",
        # DashScope 系网关对非流式调用强制要求 enable_thinking=false（否则 400）
        "disable_thinking": True,
        # DashScope qwen3-32b 的 max_tokens 上限为 16384，超出直接 400
        "max_tokens": 16384,
    },
    "gptoss120b": {
        "name": "openai/gpt-oss-120b:nitro",
        "base_url": None,            # 从 GPT_OSS_120B_BASE_URL 读取
        "api_key": None,             # 从 GPT_OSS_120B_API_KEY 读取
        "local": False,
        "health_url": None,
        "api_key_env": "GPT_OSS_120B_API_KEY",
        "base_url_env": "GPT_OSS_120B_BASE_URL",
    },
}

# 并发输出锁
_print_lock = threading.Lock()


@dataclass
class TaskResult:
    method: str
    dataset: str
    model: str
    returncode: int
    stdout: str
    stderr: str
    error_message: str | None = None


def normalize_method(value: str) -> str:
    aliases = {
        "vanilla": "vanilla",
        "dytopo": "dytopo",
        "sair": "sair",
        "tlsg": "sair",
        "pragmas_tlsg": "sair",
        "p2p_free": "p2p_free",
        "p2p_freetalk": "p2p_free",
        "autogen": "autogen",
        "autogen_groupchat": "autogen",
        "groupchat": "autogen",
        "sair_no_srac": "sair_no_srac",
        "tlsg_no_srac": "sair_no_srac",
        "sair_no_health": "sair_no_health",
        "tlsg_no_health": "sair_no_health",
        "sair_dose1": "sair_dose1",
        "sair_dose2": "sair_dose2",
        "sair_dose_r1": "sair_dose_r1",
        "sair_dose_t70": "sair_dose_t70",
        "sair_dose_r1t70": "sair_dose_r1t70",
    }
    key = value.lower().strip()
    if key not in aliases:
        raise ValueError(f"未知 method: {value!r}。可用: {', '.join(METHODS)}")
    return aliases[key]


def normalize_dataset(value: str) -> str:
    key = value.lower().strip()
    if key not in DATASETS and key != "all":
        raise ValueError(f"未知 dataset: {value!r}。可用: {', '.join(DATASETS)}")
    return key


def normalize_model(value: str) -> str:
    aliases = {
        "deepseek": "deepseek",
        "deepseek-v4-flash": "deepseek",
        "ds": "deepseek",
        "qwen3": "qwen3",
        "qwen3-8b": "qwen3",
        "gptoss120b": "gptoss120b",
        "gpt-oss-120b": "gptoss120b",
        "openai/gpt-oss-120b": "gptoss120b",
    }
    key = value.lower().strip()
    if key not in aliases:
        raise ValueError(f"未知 model: {value!r}。可用: {', '.join(MODELS)}")
    return aliases[key]


ABLATION_METHODS = {
    "sair_no_srac", "sair_no_health",
    "sair_dose1", "sair_dose2",
    "sair_dose_r1", "sair_dose_t70", "sair_dose_r1t70",
}


def config_path_for(method: str, dataset: str) -> Path:
    """根据方法名和数据集类型选择 code/reasoning 配置。"""
    prefix = METHODS[method]
    if dataset in CODE_DATASETS:
        variant = "code"
    elif dataset in REASONING_DATASETS:
        variant = "reasoning"
    else:
        raise ValueError(f"数据集 {dataset!r} 未归类到 code/reasoning")
    if method in ABLATION_METHODS:
        return ABLATION_COMPONENT_DIR / f"{prefix}_{variant}_local.yaml"
    return CONFIG_DIR / f"{prefix}_{variant}_local.yaml"


def check_local_server(health_url: str, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(health_url, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def expand_arg(value: str, all_values: Sequence[str]) -> list[str]:
    return list(all_values) if value.lower() == "all" else [value]


def build_command(
    method: str,
    dataset: str,
    model: str,
    limit: int | None,
    batch_size: int | None,
    log_level: str,
    output_dir: str | None,
    api_key: str | None,
    base_url: str | None,
    dataset_path: str | None,
) -> tuple[list[str], dict[str, str]]:
    """构造一次实验的 shell 命令和环境变量。"""
    config_path = config_path_for(method, dataset)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    model_cfg = MODELS[model]
    env = os.environ.copy()

    cmd = [
        sys.executable,
        "-m",
        "mas_topo",
        "experiment",
        str(config_path.relative_to(PROJECT_ROOT)),
        "--dataset",
        dataset,
        "--model",
        model_cfg["name"],
    ]

    # 模型 endpoint / key：所有远程模型均从 .env 中的模型专属变量读取
    if model_cfg["local"]:
        # 本地 vLLM：使用配置文件中的固定 base_url，无需 API key
        cmd.extend(["--base-url", model_cfg["base_url"]])
        if model_cfg.get("api_key"):
            cmd.extend(["--api-key", model_cfg["api_key"]])
    else:
        api_key_env = model_cfg.get("api_key_env", "OPENAI_API_KEY")
        base_url_env = model_cfg.get("base_url_env", "OPENAI_BASE_URL")
        actual_api_key = api_key or model_cfg.get("api_key") or env.get(api_key_env)
        actual_base_url = base_url or model_cfg.get("base_url") or env.get(base_url_env)
        if not actual_api_key:
            raise RuntimeError(
                f"使用 {model} 模型时必须提供 API key："
                f"通过 --api-key 传入或在 .env 中设置 {api_key_env}"
            )
        if not actual_base_url:
            raise RuntimeError(
                f"使用 {model} 模型时必须提供 base URL："
                f"通过 --base-url 传入或在 .env 中设置 {base_url_env}"
            )
        cmd.extend(["--api-key", actual_api_key, "--base-url", actual_base_url])

    if model_cfg.get("disable_thinking"):
        cmd.append("--disable-thinking")
    if model_cfg.get("max_tokens"):
        cmd.extend(["--max-tokens", str(model_cfg["max_tokens"])])

    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if output_dir:
        cmd.extend(["--output-dir", output_dir])
    if dataset_path:
        cmd.extend(["--dataset-path", dataset_path])
    cmd.extend(["--log-level", log_level])

    return cmd, env


def run_one(
    method: str,
    dataset: str,
    model: str,
    limit: int | None,
    batch_size: int | None,
    log_level: str,
    output_dir: str | None,
    api_key: str | None,
    base_url: str | None,
    dry_run: bool,
    skip_server_check: bool,
    stream: bool = False,
    allow_embedding_network: bool = False,
    dataset_path: str | None = None,
) -> TaskResult:
    """执行单个实验，返回 TaskResult。

    stream=True 时直接流式输出子进程 stdout/stderr（保留进度条等实时输出），
    不捕获内容；stream=False 时捕获输出，用于并发场景避免混叠。
    """
    method = normalize_method(method)
    dataset = normalize_dataset(dataset)
    model = normalize_model(model)

    lines: list[str] = []
    lines.extend(["=" * 70, f"[实验] method={method}, dataset={dataset}, model={model}", "=" * 70])

    model_cfg = MODELS[model]
    if model_cfg["local"] and not skip_server_check:
        health = model_cfg["health_url"]
        if not check_local_server(health):
            msg = (
                f"[警告] 本地模型 {model} 的 vLLM 服务未就绪: {health}\n"
                f"         请先启动对应服务，或使用 --skip-server-check 跳过检测。"
            )
            with _print_lock:
                print("\n".join(lines))
                print(msg, file=sys.stderr)
            return TaskResult(method, dataset, model, 1, "", msg, error_message=msg)
        lines.append(f"[检测] 本地服务就绪: {health}")

    try:
        cmd, env = build_command(
            method, dataset, model, limit, batch_size, log_level, output_dir, api_key, base_url, dataset_path
        )
    except RuntimeError as exc:
        msg = f"[错误] {exc}"
        with _print_lock:
            print("\n".join(lines))
            print(msg, file=sys.stderr)
        return TaskResult(method, dataset, model, 1, "", str(exc), error_message=str(exc))

    # 默认强制 SentenceTransformer 只读本地缓存，避免无网络环境卡住或反复请求 HuggingFace。
    # 如需更新/下载 embedding 模型，传入 --allow-embedding-network。
    env["ST_LOCAL_ONLY"] = "0" if allow_embedding_network else "1"

    # 打印命令时隐藏 API key
    display_cmd = cmd.copy()
    for i in range(len(display_cmd) - 1):
        if display_cmd[i] == "--api-key":
            display_cmd[i + 1] = "***"
    lines.append(f"[命令] {' '.join(display_cmd)}")

    if dry_run:
        lines.append("[干跑] 不执行命令")
        with _print_lock:
            print("\n".join(lines))
            print()
        return TaskResult(method, dataset, model, 0, "\n".join(lines), "")

    if stream:
        # 流式输出：保留子进程实时进度条，不捕获 stdout/stderr
        with _print_lock:
            print("\n".join(lines))
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
        with _print_lock:
            print(f"[退出码] {result.returncode}\n")
        return TaskResult(
            method=method,
            dataset=dataset,
            model=model,
            returncode=result.returncode,
            stdout="",
            stderr="",
        )

    # 非流式：捕获输出，避免多任务并发时 stdout 混叠
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines.append(f"[退出码] {result.returncode}")
    with _print_lock:
        print("\n".join(lines))
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        print()

    return TaskResult(
        method=method,
        dataset=dataset,
        model=model,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MAS_Topo 实验矩阵启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 默认跑全部方法 × 全部数据集 × 全部模型，样本级并行 5，矩阵级串行
  python scripts/run_experiment_matrix.py

  # 单个实验，样本级并行 batch=4
  python scripts/run_experiment_matrix.py --method p2p_free --dataset humaneval --model deepseek -l 20 -b 4

  # 矩阵级 4 个实验并行，每个实验内部样本串行
  python scripts/run_experiment_matrix.py --method all --dataset math --model qwen3 -j 4

  # 矩阵级 4 实验并行 + 每个实验内部 4 样本并行
  python scripts/run_experiment_matrix.py -j 4 -b 4

  # 干跑验证（OpenRouter 调用 openai/gpt-oss-120b）
  python scripts/run_experiment_matrix.py --method dytopo --dataset humaneval --model gptoss120b --dry-run
""",
    )
    parser.add_argument(
        "--method", "-M",
        default="all",
        help="方法名：vanilla / dytopo / p2p_free / sair（默认 all，只展开到 4 个主方法；消融臂 sair_no_srac / sair_no_health 需显式指定）",
    )
    parser.add_argument(
        "--dataset", "-D",
        default="all",
        help="数据集：humaneval / livecodebench / math / aime（默认 all）",
    )
    parser.add_argument(
        "--model", "-m",
        default="all",
        help="模型：deepseek / qwen3 / gptoss120b（默认 all）",
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="覆盖 dataset.limit，跑多少条样本（默认全部）",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="单个实验内部的样本级并发数（默认 5）",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="日志级别",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="自定义输出目录（默认自动生成）",
    )
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="覆盖数据集文件路径（保留 --dataset 的 loader 和 metric）",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="覆盖 .env 中模型对应的 API key 环境变量",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="覆盖 .env 中模型对应的 base URL 环境变量",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印命令，不执行",
    )
    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="跳过本地 vLLM 服务健康检查",
    )
    parser.add_argument(
        "--allow-embedding-network",
        action="store_true",
        help="允许 SentenceTransformer 从网络下载/更新 embedding 模型（默认只用本地缓存，避免无网络时卡住）",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=1,
        help="矩阵级实验并发数：同时跑多少个（method, dataset, model）组合（默认 1）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    methods = expand_arg(args.method, MAIN_METHODS)
    datasets = expand_arg(args.dataset, DATASETS)
    models = expand_arg(args.model, list(MODELS.keys()))

    # 先做规范化校验
    methods = [normalize_method(m) for m in methods]
    datasets = [normalize_dataset(d) for d in datasets]
    models = [normalize_model(m) for m in models]

    total = len(methods) * len(datasets) * len(models)
    print(f"[矩阵] 共 {total} 个实验: methods={methods}, datasets={datasets}, models={models}, workers={args.workers}")
    print()

    tasks = [
        (method, dataset, model)
        for method in methods
        for dataset in datasets
        for model in models
    ]

    results: list[TaskResult] = []
    stream = args.workers <= 1
    if args.workers <= 1:
        # 顺序执行：默认流式输出，保留底层进度条
        for method, dataset, model in tasks:
            results.append(
                run_one(
                    method=method,
                    dataset=dataset,
                    model=model,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    log_level=args.log_level,
                    output_dir=args.output_dir,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    dry_run=args.dry_run,
                    skip_server_check=args.skip_server_check,
                    stream=stream,
                    allow_embedding_network=args.allow_embedding_network,
                    dataset_path=args.dataset_path,
                )
            )
    else:
        # 并发执行：捕获输出避免混叠
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_task = {
                executor.submit(
                    run_one,
                    method=method,
                    dataset=dataset,
                    model=model,
                    limit=args.limit,
                    batch_size=args.batch_size,
                    log_level=args.log_level,
                    output_dir=args.output_dir,
                    api_key=args.api_key,
                    base_url=args.base_url,
                    dry_run=args.dry_run,
                    skip_server_check=args.skip_server_check,
                    stream=False,
                    allow_embedding_network=args.allow_embedding_network,
                    dataset_path=args.dataset_path,
                ): (method, dataset, model)
                for method, dataset, model in tasks
            }
            for future in as_completed(future_to_task):
                try:
                    results.append(future.result())
                except Exception as exc:
                    method, dataset, model = future_to_task[future]
                    msg = f"[异常] {method}/{dataset}/{model}: {exc}"
                    with _print_lock:
                        print(msg, file=sys.stderr)
                    results.append(
                        TaskResult(method, dataset, model, 1, "", str(exc), error_message=str(exc))
                    )

    failed = [(r.method, r.dataset, r.model) for r in results if r.returncode != 0]

    print("=" * 70)
    print(f"[完成] 成功 {total - len(failed)} / 失败 {len(failed)} / 总计 {total}")
    if failed:
        print("[失败组合]")
        for method, dataset, model in failed:
            print(f"  - {method} / {dataset} / {model}")
    print("=" * 70)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
