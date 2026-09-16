"""PragmaMAS 每轮拓扑与实验细节可视化工具。

用法:
    python visualization/visualize_rounds.py experiment_results/<run>/trace/sample_000/trace.json
    python visualization/visualize_rounds.py experiment_results/<run>/trace/sample_000/trace.json \
        --output-dir figures/sample_000_rounds \
        --plots topology,communication,metrics,activity \
        --format pdf
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


VALID_PLOTS = ["topology", "communication", "metrics", "activity"]
AGENT_COLORS = {
    "Coordinator": "#E74C3C",
    "Developer": "#3498DB",
    "Researcher": "#2ECC71",
    "Designer": "#F39C12",
    "Tester": "#9B59B6",
}
DEFAULT_AGENT_COLOR = "#7F8C8D"
ILLOCUTION_COLORS = {
    "Directive": "#C0392B",
    "Assertive": "#2980B9",
    "Commissive": "#27AE60",
    "Expressive": "#E67E22",
    "Declarative": "#8E44AD",
    "Tool": "#16A085",
}
DEFAULT_ILLOCUTION_COLOR = "#7F8C8D"


def load_trace(trace_path: str) -> dict:
    """加载 trace.json。"""
    with open(trace_path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_plots(plots_arg: str) -> list[str]:
    """解析 --plots 参数。"""
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


def parse_rounds(rounds_arg: str, all_round_nums: list[int]) -> list[int]:
    """解析 --rounds 参数。"""
    if rounds_arg == "all":
        return all_round_nums
    selected = []
    for part in rounds_arg.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            rn = int(part)
        except ValueError:
            raise ValueError(f"Invalid round number: {part}")
        if rn not in all_round_nums:
            raise ValueError(f"Round {rn} not found in trace")
        selected.append(rn)
    return selected


def _make_pos(layout: str, G: nx.DiGraph, **kwargs):
    """根据布局名称生成节点位置。"""
    if layout == "circular":
        return nx.circular_layout(G, scale=1.8, **kwargs)
    if layout == "spring":
        return nx.spring_layout(G, seed=42, k=2.0, **kwargs)
    if layout == "shell":
        return nx.shell_layout(G, **kwargs)
    if layout == "kamada":
        return nx.kamada_kawai_layout(G, **kwargs)
    return nx.spring_layout(G, seed=42, k=2.0, **kwargs)


def _agent_color(name: str) -> str:
    """根据 Agent 名称推断角色颜色。"""
    for role, color in AGENT_COLORS.items():
        if role.lower() in name.lower():
            return color
    return DEFAULT_AGENT_COLOR


def safe_plot(name: str, plot_fn, round_data: dict, output_dir: Path, fmt: str, layout: str):
    """安全调用绘图函数；失败时打印警告但继续。"""
    try:
        plot_fn(round_data, output_dir, fmt, layout)
    except Exception as exc:
        print(
            f"Warning: plot '{name}' failed for round {round_data.get('round_num')}: {exc}",
            file=sys.stderr,
        )


def plot_topology(round_data: dict, output_dir: Path, fmt: str, layout: str):
    """绘制每轮拓扑结构图。"""
    adjacency = round_data.get("adjacency_list")
    if not adjacency:
        return

    G = nx.DiGraph()
    for src, tgts in adjacency.items():
        G.add_node(src)
        for tgt in tgts:
            G.add_edge(src, tgt)

    pos = _make_pos(layout, G)
    fig, ax = plt.subplots(figsize=(8, 7))
    node_list = sorted(G.nodes())
    node_colors = [_agent_color(n) for n in node_list]

    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list, node_color=node_colors,
        edgecolors="#2C3E50", node_size=3000, linewidths=2, ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, labels={n: n for n in node_list},
        font_size=10, font_weight="bold", font_color="#1A252F", ax=ax,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=list(G.edges()), edge_color="#95A5A6",
        arrowsize=18, arrowstyle="-|>", width=1.5, ax=ax, node_size=3000,
    )

    strategy = round_data.get("topology_strategy", "Unknown")
    ax.set_title(f"Round {round_data['round_num']} Topology — {strategy}", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_dir / f"topology_round_{round_data['round_num']:03d}.{fmt}", dpi=200)
    plt.close(fig)


def plot_communication(round_data: dict, output_dir: Path, fmt: str, layout: str):
    """绘制每轮实际通信图。"""
    edges = round_data.get("edges", [])
    if not edges:
        return

    G = nx.DiGraph()
    for name in round_data.get("agents", {}).keys():
        G.add_node(name)
    for e in edges:
        G.add_node(e["source"])
        G.add_node(e["target"])
        G.add_edge(e["source"], e["target"], **e)

    pos = _make_pos(layout, G)
    fig, ax = plt.subplots(figsize=(9, 8))
    node_list = sorted(G.nodes())
    node_colors = [_agent_color(n) for n in node_list]

    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list, node_color=node_colors,
        edgecolors="#2C3E50", node_size=3000, linewidths=2, ax=ax,
    )
    nx.draw_networkx_labels(
        G, pos, labels={n: n for n in node_list},
        font_size=10, font_weight="bold", font_color="#1A252F", ax=ax,
    )

    edge_groups: dict[tuple, list[dict]] = {}
    for src, tgt, data in G.edges(data=True):
        edge_groups.setdefault((src, tgt), []).append(data)

    for (src, tgt), group in edge_groups.items():
        n = len(group)
        for i, e_data in enumerate(group):
            cat = e_data.get("metadata", {}).get("illocution_category", "Unknown")
            color = ILLOCUTION_COLORS.get(cat, DEFAULT_ILLOCUTION_COLOR)
            channel = e_data.get("channel", "")
            style = "dashed" if "public" in channel.lower() else "solid"
            rad = 0.12 + (i - n / 2) * 0.08 if n > 1 else 0.12
            nx.draw_networkx_edges(
                G, pos, edgelist=[(src, tgt)], edge_color=[color],
                width=2.5, arrowsize=20, arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}", style=style,
                ax=ax, node_size=3000, alpha=0.9,
            )

    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=3, label=cat)
        for cat, c in ILLOCUTION_COLORS.items()
    ]
    legend_elements.append(plt.Line2D([0], [0], color="#555555", lw=2, linestyle="-", label="private"))
    legend_elements.append(plt.Line2D([0], [0], color="#555555", lw=2, linestyle="--", label="public"))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              title="Illocution / Channel", title_fontsize=10, framealpha=0.95)

    strategy = round_data.get("routing_strategy", "Unknown")
    ax.set_title(f"Round {round_data['round_num']} Communication — {strategy}", fontsize=14, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_dir / f"communication_round_{round_data['round_num']:03d}.{fmt}", dpi=200)
    plt.close(fig)


def plot_metrics(round_data: dict, output_dir: Path, fmt: str, layout: str):
    """绘制每轮指标卡片。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    density = round_data.get("topology_metadata", {}).get("density")
    density_str = f"{density:.3f}" if isinstance(density, (int, float)) else "-"
    elapsed = round_data.get("elapsed_time", "-")
    elapsed_str = f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) else "-"

    lines = [
        f"Round {round_data.get('round_num', '-')}",
        "",
        f"Agents:            {round_data.get('agent_count', '-')}",
        f"Edges:             {round_data.get('edge_count', '-')}",
        f"Density:           {density_str}",
        f"Prompt tokens:     {round_data.get('prompt_tokens', '-')}",
        f"Completion tokens: {round_data.get('completion_tokens', '-')}",
        f"Elapsed time:      {elapsed_str}",
    ]

    decision = round_data.get("decision")
    if decision:
        status = decision.get("status", "unknown")
        lines.append(f"Decision:          {status}")

    fig.text(0.15, 0.55, "\n".join(lines), fontsize=13, family="monospace",
             verticalalignment="center", color="#1A252F")
    ax.set_title("Round Metrics", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(output_dir / f"metrics_round_{round_data['round_num']:03d}.{fmt}", dpi=200)
    plt.close(fig)


def plot_activity(round_data: dict, output_dir: Path, fmt: str, layout: str):
    """绘制每轮 Agent 活跃度与语旨分布图。"""
    edges = round_data.get("edges", [])
    agents = sorted(round_data.get("agents", {}).keys())
    if not agents:
        return

    sent = Counter()
    received = Counter()
    illocution = Counter()
    for e in edges:
        sent[e.get("source", "")] += 1
        received[e.get("target", "")] += 1
        cat = e.get("metadata", {}).get("illocution_category", "Unknown")
        illocution[cat] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(agents))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], [sent[a] for a in agents], width, label="Sent", color="#3498DB")
    ax1.bar([i + width / 2 for i in x], [received[a] for a in agents], width, label="Received", color="#2ECC71")
    ax1.set_xticks(x)
    ax1.set_xticklabels(agents)
    ax1.set_xlabel("Agent")
    ax1.set_ylabel("Message Count")
    ax1.set_title("Agent Message Activity")
    ax1.legend()

    if illocution:
        labels, values = zip(*sorted(illocution.items(), key=lambda x: x[1], reverse=True))
        colors = [ILLOCUTION_COLORS.get(l, DEFAULT_ILLOCUTION_COLOR) for l in labels]
        ax2.bar(labels, values, color=colors)
        ax2.set_xlabel("Illocution")
        ax2.set_ylabel("Count")
        ax2.set_title("Illocution Distribution")

    fig.suptitle(f"Round {round_data['round_num']} Activity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / f"activity_round_{round_data['round_num']:03d}.{fmt}", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize PragmaMAS per-round topology and details")
    parser.add_argument("trace_json", help="Path to trace.json")
    parser.add_argument("--output-dir", default=None, help="Directory to save figures")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"],
                        help="Output figure format")
    parser.add_argument("--plots", default="all",
                        help="Comma-separated plot names: topology,communication,metrics,activity")
    parser.add_argument("--rounds", default="all",
                        help="Comma-separated round numbers to visualize; default all")
    parser.add_argument("--layout", default="spring",
                        choices=["circular", "spring", "shell", "kamada"],
                        help="Node layout algorithm")
    args = parser.parse_args()

    trace_path = Path(args.trace_json)
    if not trace_path.exists():
        parser.error(f"Trace file not found: {trace_path}")
        return

    data = load_trace(str(trace_path))
    rounds = data.get("rounds", [])
    if not rounds:
        print("No rounds found in trace.")
        return

    all_round_nums = [r["round_num"] for r in rounds]
    try:
        selected_round_nums = parse_rounds(args.rounds, all_round_nums)
        plot_names = parse_plots(args.plots)
    except ValueError as exc:
        parser.error(str(exc))
        return

    selected_rounds = [r for r in rounds if r["round_num"] in selected_round_nums]

    output_dir = Path(args.output_dir) if args.output_dir else trace_path.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_registry = {
        "topology": plot_topology,
        "communication": plot_communication,
        "metrics": plot_metrics,
        "activity": plot_activity,
    }

    for r in selected_rounds:
        for name in plot_names:
            safe_plot(name, plot_registry[name], r, output_dir, args.format, args.layout)

    print(f"Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
