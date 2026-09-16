"""可视化实验指标。

用法:
    python visualization/visualize_metrics.py experiment_results/<run>/
    python visualization/visualize_metrics.py --metrics experiment_results/<run>/metrics.jsonl --output-dir experiment_results/<run>/figures
    python visualization/visualize_metrics.py experiment_results/<run>/ --format pdf --plots accuracy,rounds,tokens
    python visualization/visualize_metrics.py experiment_results/<run>/ --offset 10 --num 5

支持 --plots: all, accuracy, rounds, tokens, topology, activation, illocution, channel
"""

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


VALID_PLOTS = ["accuracy", "rounds", "tokens", "topology", "activation", "illocution", "channel"]


def resolve_paths(experiment_dir: str | None, metrics: str | None, output_dir: str | None) -> tuple[str, str]:
    """解析指标文件路径与输出目录路径。"""
    if experiment_dir is not None:
        exp_path = Path(experiment_dir)
        if not exp_path.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        if not exp_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {experiment_dir}")

        candidates = [exp_path / "metrics.jsonl", exp_path / "metrics.csv"]
        metrics_path = None
        for cand in candidates:
            if cand.exists():
                metrics_path = str(cand)
                break
        if metrics_path is None:
            raise FileNotFoundError(
                f"No metrics file found in {experiment_dir}. "
                "Expected one of: metrics.jsonl, metrics.csv"
            )

        out = output_dir if output_dir is not None else str(exp_path / "figures")
        return metrics_path, out

    if metrics is not None:
        if output_dir is None:
            raise ValueError(
                "When using --metrics without an experiment_dir, "
                "you must also specify --output-dir."
            )
        if not Path(metrics).exists():
            raise FileNotFoundError(f"Metrics file not found: {metrics}")
        return metrics, output_dir

    raise ValueError(
        "Please provide either an experiment_dir positional argument "
        "or --metrics (with --output-dir)."
    )


def parse_plots(plots_arg: str) -> list[str]:
    """解析 --plots 参数为图表名称列表。"""
    plots_arg = plots_arg.strip().lower()
    if plots_arg == "all":
        return list(VALID_PLOTS)
    selected = [p.strip() for p in plots_arg.split(",") if p.strip()]
    invalid = [p for p in selected if p not in VALID_PLOTS]
    if invalid:
        raise ValueError(
            f"Invalid plot name(s): {invalid}. "
            f"Valid options: all or {', '.join(VALID_PLOTS)}"
        )
    return selected


def filter_samples(df_rounds: pd.DataFrame, offset: int, num: int) -> pd.DataFrame:
    """按 sample_index 排序后取 [offset, offset+num) 区间的样例。"""
    if df_rounds.empty:
        return df_rounds

    unique_indices = sorted(df_rounds["sample_index"].unique())
    total = len(unique_indices)

    if offset < 0:
        raise ValueError(f"--offset must be >= 0, got {offset}")
    if offset >= total:
        raise ValueError(
            f"--offset ({offset}) exceeds the number of available samples ({total})"
        )
    if num == 0:
        return df_rounds.iloc[0:0]
    if num < -1:
        raise ValueError(f"--num must be >= -1, got {num}")

    end = total if num == -1 else min(offset + num, total)
    if num != -1 and end < offset + num:
        print(
            f"Warning: --offset {offset} + --num {num} exceeds sample count {total}, "
            f"clamping to {end - offset} samples.",
            file=sys.stderr,
        )

    selected = unique_indices[offset:end]
    return df_rounds[df_rounds["sample_index"].isin(selected)].copy()


def load_metrics(metrics_path: str) -> pd.DataFrame:
    """加载指标文件，返回 round-level DataFrame。"""
    path = Path(metrics_path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    if path.suffix == ".jsonl":
        runs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                runs.append(json.loads(line))
        # 将 runs 转为 round-level 行
        rows = []
        for run in runs:
            sample_index = run.get("sample_index", -1)
            base = {
                "sample_index": sample_index,
                "task": run.get("task", ""),
                "total_rounds": run.get("total_rounds", 0),
                "is_completed": run.get("is_completed", False),
                "is_correct": run.get("is_correct", False),
                "total_cost": run.get("total_cost", 0.0),
                "total_prompt_tokens": run.get("total_prompt_tokens", 0),
                "total_completion_tokens": run.get("total_completion_tokens", 0),
                "elapsed_time": run.get("elapsed_time", 0.0),
                "topology_strategy": run.get("topology_strategy", ""),
                "routing_strategy": run.get("routing_strategy", ""),
                "decision_strategy": run.get("decision_strategy", ""),
            }
            rounds = run.get("rounds", [])
            if rounds:
                for rm in rounds:
                    row = dict(base)
                    row.update(rm)
                    rows.append(row)
            else:
                rows.append(base)
        return pd.DataFrame(rows)

    # CSV 每行已经是一轮
    df = pd.read_csv(metrics_path)
    # active_agents 是 ";" 分隔的字符串，重新解析为列表
    if "active_agents" in df.columns:
        df["active_agents"] = df["active_agents"].fillna("").apply(
            lambda x: [a.strip() for a in str(x).split(";") if a.strip()]
        )
    # illocution_counts / channel_counts 是 JSON 字符串
    for col in ["illocution_counts", "channel_counts"]:
        if col in df.columns:
            df[col] = df[col].fillna("{}").apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.strip() else {}
            )
    return df


def get_run_level(df: pd.DataFrame) -> pd.DataFrame:
    """从 round-level DataFrame 聚合出 run-level DataFrame。"""
    return df.groupby("sample_index").first().reset_index()


def plot_accuracy_overview(df_runs: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """准确率概览。"""
    total = len(df_runs)
    completed = df_runs["is_completed"].sum()
    correct = df_runs["is_correct"].sum()
    accuracy = correct / total if total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Total", "Completed", "Correct"], [total, completed, correct], color=["#888888", "#4C78A8", "#54A24B"])
    ax.set_ylabel("Number of Samples")
    ax.set_title(f"Accuracy Overview (Accuracy = {accuracy:.2%})")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_dir / f"accuracy_overview.{fmt}", dpi=150)
    plt.close(fig)


def plot_rounds_distribution(df_runs: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """轮数分布。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df_runs["total_rounds"], bins=range(1, int(df_runs["total_rounds"].max()) + 3), color="#4C78A8", edgecolor="white")
    ax.set_xlabel("Total Rounds")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Distribution of Total Rounds")
    fig.tight_layout()
    fig.savefig(output_dir / f"rounds_distribution.{fmt}", dpi=150)
    plt.close(fig)


def plot_token_usage(df_runs: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """Token 使用情况。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df_runs["total_prompt_tokens"], df_runs["total_completion_tokens"],
               c=df_runs["is_correct"].map({True: "#54A24B", False: "#E45756"}), alpha=0.6)
    ax.set_xlabel("Prompt Tokens")
    ax.set_ylabel("Completion Tokens")
    ax.set_title("Token Usage per Sample (Green=Correct, Red=Wrong)")
    fig.tight_layout()
    fig.savefig(output_dir / f"token_usage.{fmt}", dpi=150)
    plt.close(fig)


def plot_topology_evolution(df_rounds: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """拓扑指标随轮次演化。"""
    if df_rounds.empty:
        return
    grouped = df_rounds.groupby("round_num").agg({
        "graph_density": "mean",
        "num_messages": "mean",
        "num_edges": "mean",
    }).reset_index()

    fig, ax1 = plt.subplots(figsize=(7, 4))
    color = "#4C78A8"
    ax1.set_xlabel("Round Number")
    ax1.set_ylabel("Graph Density", color=color)
    ax1.plot(grouped["round_num"], grouped["graph_density"], color=color, marker="o", label="Density")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "#E45756"
    ax2.set_ylabel("Avg Messages", color=color)
    ax2.plot(grouped["round_num"], grouped["num_messages"], color=color, marker="s", linestyle="--", label="Messages")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.suptitle("Topology Evolution over Rounds")
    fig.tight_layout()
    fig.savefig(output_dir / f"topology_evolution.{fmt}", dpi=150)
    plt.close(fig)


def plot_activation_heatmap(df_rounds: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """Agent 激活热图。"""
    if "active_agents" not in df_rounds.columns or df_rounds.empty:
        return
    all_agents = sorted(set(agent for agents in df_rounds["active_agents"] for agent in agents if agent))
    if not all_agents:
        return

    # 每轮每个 Agent 被激活的次数
    counts = {agent: Counter() for agent in all_agents}
    for _, row in df_rounds.iterrows():
        rn = row["round_num"]
        for agent in row["active_agents"]:
            if agent in counts:
                counts[agent][rn] += 1

    max_round = int(df_rounds["round_num"].max())
    matrix = []
    for agent in all_agents:
        matrix.append([counts[agent].get(rn, 0) for rn in range(1, max_round + 1)])

    fig, ax = plt.subplots(figsize=(max(6, max_round * 0.5), max(4, len(all_agents) * 0.5)))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xticks(range(max_round))
    ax.set_xticklabels(range(1, max_round + 1))
    ax.set_yticks(range(len(all_agents)))
    ax.set_yticklabels(all_agents)
    ax.set_xlabel("Round Number")
    ax.set_ylabel("Agent")
    ax.set_title("Agent Activation Heatmap")
    fig.colorbar(im, ax=ax, label="Activation Count")
    fig.tight_layout()
    fig.savefig(output_dir / f"activation_heatmap.{fmt}", dpi=150)
    plt.close(fig)


def plot_illocution_distribution(df_rounds: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """语旨类别分布。"""
    if "illocution_counts" not in df_rounds.columns or df_rounds.empty:
        return
    total = Counter()
    for counts in df_rounds["illocution_counts"]:
        if isinstance(counts, dict):
            total.update(counts)
    if not total:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    labels, values = zip(*sorted(total.items(), key=lambda x: x[1], reverse=True))
    ax.bar(labels, values, color="#4C78A8")
    ax.set_xlabel("Illocution")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Illocutionary Acts")
    fig.tight_layout()
    fig.savefig(output_dir / f"illocution_distribution.{fmt}", dpi=150)
    plt.close(fig)


def plot_channel_distribution(df_rounds: pd.DataFrame, output_dir: Path, fmt: str = "png"):
    """消息信道分布。"""
    if "channel_counts" not in df_rounds.columns or df_rounds.empty:
        return
    total = Counter()
    for counts in df_rounds["channel_counts"]:
        if isinstance(counts, dict):
            total.update(counts)
    if not total:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    labels, values = zip(*sorted(total.items(), key=lambda x: x[1], reverse=True))
    ax.bar(labels, values, color="#E45756")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Message Channels")
    fig.tight_layout()
    fig.savefig(output_dir / f"channel_distribution.{fmt}", dpi=150)
    plt.close(fig)


def print_summary(df_runs: pd.DataFrame, total_samples: int | None = None):
    """在终端打印运行摘要统计。"""
    total = len(df_runs)
    if total == 0:
        print("Summary: no runs found.")
        return
    completed = int(df_runs["is_completed"].sum())
    correct = int(df_runs["is_correct"].sum())
    sample_label = f"{total} / {total_samples}" if total_samples is not None else str(total)
    print("Summary")
    print(f"  Samples:                 {sample_label}")
    print(f"  Completed:               {completed} ({completed / total:.1%})")
    print(f"  Correct:                 {correct} ({correct / total:.1%})")
    if "total_rounds" in df_runs.columns:
        print(f"  Avg rounds:              {df_runs['total_rounds'].mean():.2f}")
    if "total_prompt_tokens" in df_runs.columns:
        print(f"  Avg prompt tokens:       {df_runs['total_prompt_tokens'].mean():,.0f}")
    if "total_completion_tokens" in df_runs.columns:
        print(f"  Avg completion tokens:   {df_runs['total_completion_tokens'].mean():,.0f}")


def safe_plot(name: str, plot_fn, df_runs: pd.DataFrame, df_rounds: pd.DataFrame, output_dir: Path, fmt: str):
    """安全调用绘图函数；失败时打印警告但继续。"""
    try:
        sig = inspect.signature(plot_fn)
        params = list(sig.parameters.keys())
        if "df_rounds" in params:
            plot_fn(df_rounds, output_dir, fmt)
        else:
            plot_fn(df_runs, output_dir, fmt)
    except Exception as exc:
        print(f"Warning: plot '{name}' failed: {exc}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Visualize PragmaMAS experiment metrics")
    parser.add_argument("experiment_dir", nargs="?", default=None,
                        help="Experiment directory containing metrics.jsonl or metrics.csv")
    parser.add_argument("--metrics", default=None, help="Path to metrics.jsonl or metrics.csv")
    parser.add_argument("--output-dir", default=None, help="Directory to save figures")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output figure format")
    parser.add_argument("--plots", default="all",
                        help="Comma-separated plot names: all or accuracy,rounds,tokens,topology,activation,illocution,channel")
    parser.add_argument("--offset", type=int, default=0,
                        help="0-based offset of the first sample to visualize")
    parser.add_argument("--num", type=int, default=-1,
                        help="Number of samples to visualize; -1 means all from offset")
    args = parser.parse_args()

    try:
        metrics_path, output_dir = resolve_paths(args.experiment_dir, args.metrics, args.output_dir)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        parser.error(str(exc))
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_rounds = load_metrics(metrics_path)
    if df_rounds.empty:
        print("No metrics data to visualize.")
        return

    total_samples = df_rounds["sample_index"].nunique()
    try:
        df_rounds = filter_samples(df_rounds, args.offset, args.num)
    except ValueError as exc:
        parser.error(str(exc))
        return
    if df_rounds.empty:
        print("No samples selected after filtering.")
        return

    df_runs = get_run_level(df_rounds)
    plot_names = parse_plots(args.plots)

    plot_registry = {
        "accuracy": plot_accuracy_overview,
        "rounds": plot_rounds_distribution,
        "tokens": plot_token_usage,
        "topology": plot_topology_evolution,
        "activation": plot_activation_heatmap,
        "illocution": plot_illocution_distribution,
        "channel": plot_channel_distribution,
    }

    for name in plot_names:
        safe_plot(name, plot_registry[name], df_runs, df_rounds, output_dir, args.format)

    print_summary(df_runs, total_samples)
    print(f"\nFigures saved to: {output_dir}")
    for p in sorted(output_dir.glob(f"*.{args.format}")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
