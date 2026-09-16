#!/usr/bin/env python3
"""PragmaMAS 通信图谱可视化工具。

将实验 trace 中各轮次的 private_routing 绘制成有向图，
用颜色区分 5 种语旨类别（Directive/Assertive/Commissive/Expressive/Declarative）。

支持三种可视化模式：
1. 逐轮动态图（默认）— 每轮一张静态图，可选合成 GIF
2. 聚合汇总图（--aggregate）— 全轮次合并，边粗细 = 通信频次
3. 时间展开图（--time-unfolded）— 节点按轮次展开为 DAG，展示信息流演化

Usage:
    python visualization/visualize_routing.py \
        experiment_results/.../trace/sample_000/trace.json \
        --output-dir figures/sample_000 \
        --gif --aggregate --time-unfolded

Output:
    - round_01.png ~ round_N.png       每轮静态图
    - routing_animation.gif            动态合成（--gif 时）
    - aggregate.png                    全轮次聚合图（--aggregate 时）
    - time_unfolded.png                时间展开 DAG（--time-unfolded 时）
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

# ── 语旨 → 颜色映射（学术配色）─────────────────────────────
ILLOCUTION_COLORS = {
    "Directive":   "#C0392B",   # 深红 — 指令
    "Assertive":   "#2980B9",   # 深蓝 — 断言
    "Commissive":  "#27AE60",   # 深绿 — 承诺
    "Expressive":  "#E67E22",   # 深橙 — 表达
    "Declarative": "#8E44AD",   # 深紫 — 宣告
    "Tool":        "#16A085",   # 深青 — 工具调用
}

DEFAULT_COLOR = "#7F8C8D"
NODE_COLOR = "#F8F9FA"
NODE_EDGE_COLOR = "#2C3E50"
INACTIVE_NODE_COLOR = "#ECEFF1"
INACTIVE_EDGE_COLOR = "#B0BEC5"


def load_trace(trace_path: str) -> dict:
    with open(trace_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_private_edges(round_data: dict) -> list[dict]:
    edges = []
    for e in round_data.get("edges", []):
        channel = e.get("channel", "")
        if "private" in channel.lower():
            edges.append({
                "source": e["source"],
                "target": e["target"],
                "category": e.get("metadata", {}).get("illocution_category", "Unknown"),
                "content": e.get("metadata", {}).get("illocution_content", ""),
            })
    return edges


def build_agent_set(rounds: list[dict]) -> set[str]:
    agents = set()
    for r in rounds:
        for name in r.get("agents", {}).keys():
            agents.add(name)
        for e in r.get("edges", []):
            agents.add(e.get("source"))
            agents.add(e.get("target"))
    return agents


def _make_pos(layout: str, G: nx.DiGraph, **kwargs):
    if layout == "circular":
        return nx.circular_layout(G, scale=1.8, **kwargs)
    if layout == "spring":
        return nx.spring_layout(G, seed=42, k=2.0, **kwargs)
    if layout == "shell":
        return nx.shell_layout(G, **kwargs)
    if layout == "kamada":
        return nx.kamada_kawai_layout(G, **kwargs)
    return nx.circular_layout(G, scale=1.8, **kwargs)


def draw_round(
    pos: dict,
    round_num: int,
    edges: list[dict],
    all_agents: set[str],
    output_path: Path,
    figsize: tuple = (10, 8),
    dpi: int = 200,
    task: str = "",
):
    """绘制单轮通信图。"""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    G = nx.DiGraph()
    G.add_nodes_from(sorted(all_agents))

    node_list = sorted(all_agents)
    node_colors = []
    node_edge_colors = []

    active = {e["source"] for e in edges} | {e["target"] for e in edges}

    for n in node_list:
        if n in active:
            node_colors.append(NODE_COLOR)
            node_edge_colors.append(NODE_EDGE_COLOR)
        else:
            node_colors.append(INACTIVE_NODE_COLOR)
            node_edge_colors.append(INACTIVE_EDGE_COLOR)

    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        node_size=3200,
        linewidths=2.2,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G, pos, labels={n: n for n in node_list},
        font_size=11, font_weight="bold", font_color="#1A252F",
        ax=ax,
    )

    # 边按 (source, target, category) 分组，避免同一对节点多条边完全重叠
    edge_groups: dict[tuple, list[dict]] = {}
    for e in edges:
        key = (e["source"], e["target"])
        edge_groups.setdefault(key, []).append(e)

    for (src, tgt), group in edge_groups.items():
        n = len(group)
        for i, e in enumerate(group):
            cat = e["category"]
            color = ILLOCUTION_COLORS.get(cat, DEFAULT_COLOR)
            # 多条边时用不同弧度错开
            rad = 0.12 + (i - n / 2) * 0.08 if n > 1 else 0.12

            nx.draw_networkx_edges(
                G, pos, edgelist=[(src, tgt)],
                edge_color=[color],
                width=2.8,
                arrowsize=22,
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad}",
                ax=ax,
                node_size=3200,
                alpha=0.92,
            )

        # 边标签：若多条同类别则只显示一次
        cats = [e["category"] for e in group]
        label = cats[0] if len(set(cats)) == 1 else "/".join(dict.fromkeys(cats))
        # 手动放置标签到弧线中点附近
        mid_x = (pos[src][0] + pos[tgt][0]) / 2
        mid_y = (pos[src][1] + pos[tgt][1]) / 2
        # 根据弧度偏移
        offset = 0.08 * n if n > 1 else 0.05
        ax.text(
            mid_x, mid_y + offset, label,
            fontsize=7.5, color="#444444", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.85),
            zorder=5,
        )

    # 图例
    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=3.5, label=cat)
        for cat, c in ILLOCUTION_COLORS.items()
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=9.5,
        title="Illocutionary Force",
        title_fontsize=10.5,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    title = f"Round {round_num}  —  Private Routing Graph"
    if task:
        task_short = task.replace("\n", " ")[:60] + "..." if len(task) > 60 else task
        title += f"\n{task_short}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color="#1A252F")

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)


def draw_aggregate(
    pos: dict,
    all_edges: list[dict],
    all_agents: set[str],
    output_path: Path,
    figsize: tuple = (10, 8),
    dpi: int = 200,
    task: str = "",
):
    """绘制全轮次聚合图。

    边粗细 = 出现频次，边颜色 = 主导语旨（出现最多的类别）。
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    G = nx.DiGraph()
    G.add_nodes_from(sorted(all_agents))

    # 统计 (src, tgt) → 频次 及 各类别计数
    edge_stats: dict[tuple, Counter] = {}
    for e in all_edges:
        key = (e["source"], e["target"])
        edge_stats.setdefault(key, Counter())[e["category"]] += 1

    node_list = sorted(all_agents)
    active = set()
    for (src, tgt) in edge_stats:
        active.add(src)
        active.add(tgt)

    node_colors = []
    node_edge_colors = []
    for n in node_list:
        if n in active:
            node_colors.append(NODE_COLOR)
            node_edge_colors.append(NODE_EDGE_COLOR)
        else:
            node_colors.append(INACTIVE_NODE_COLOR)
            node_edge_colors.append(INACTIVE_EDGE_COLOR)

    nx.draw_networkx_nodes(
        G, pos, nodelist=node_list,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        node_size=3200,
        linewidths=2.2,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G, pos, labels={n: n for n in node_list},
        font_size=11, font_weight="bold", font_color="#1A252F",
        ax=ax,
    )

    # 绘制聚合边
    max_count = max((sum(c.values()) for c in edge_stats.values()), default=1)
    for (src, tgt), counter in edge_stats.items():
        total = sum(counter.values())
        dominant_cat = counter.most_common(1)[0][0]
        color = ILLOCUTION_COLORS.get(dominant_cat, DEFAULT_COLOR)
        width = 1.5 + 4.0 * (total / max_count)
        alpha = 0.5 + 0.45 * (total / max_count)

        nx.draw_networkx_edges(
            G, pos, edgelist=[(src, tgt)],
            edge_color=[color],
            width=width,
            arrowsize=22,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.12",
            ax=ax,
            node_size=3200,
            alpha=alpha,
        )

        mid_x = (pos[src][0] + pos[tgt][0]) / 2
        mid_y = (pos[src][1] + pos[tgt][1]) / 2
        label = f"{total}× {dominant_cat}"
        ax.text(
            mid_x, mid_y + 0.06, label,
            fontsize=8, color="#333333", ha="center", va="bottom",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.9),
            zorder=5,
        )

    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=3.5, label=cat)
        for cat, c in ILLOCUTION_COLORS.items()
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=9.5,
        title="Illocutionary Force",
        title_fontsize=10.5,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    title = "Aggregate Private Routing Graph  —  All Rounds"
    if task:
        task_short = task.replace("\n", " ")[:60] + "..." if len(task) > 60 else task
        title += f"\n{task_short}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color="#1A252F")

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)


def draw_time_unfolded(
    rounds: list[dict],
    all_agents: set[str],
    output_path: Path,
    figsize: tuple = (14, 10),
    dpi: int = 200,
    task: str = "",
):
    """绘制时间展开 DAG。

    节点命名为 "Agent@R1", "Agent@R2" ...
    边连接跨轮次的通信关系。
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    G = nx.DiGraph()
    pos = {}
    node_colors = {}
    agent_x = {name: i for i, name in enumerate(sorted(all_agents))}

    # 创建每轮节点
    for r in rounds:
        rn = r["round_num"]
        for name in sorted(all_agents):
            node_id = f"{name}@R{rn}"
            G.add_node(node_id)
            pos[node_id] = (agent_x[name], -rn)
            node_colors[node_id] = NODE_COLOR

    # 添加边：本轮 src → 本轮 tgt（跨轮次链接到下一轮同 agent）
    edge_list = []
    edge_colors = []
    edge_widths = []
    edge_labels_data = {}

    for r in rounds:
        rn = r["round_num"]
        edges = extract_private_edges(r)
        for e in edges:
            src_node = f"{e['source']}@R{rn}"
            tgt_node = f"{e['target']}@R{rn}"
            if src_node in G and tgt_node in G:
                G.add_edge(src_node, tgt_node)
                edge_list.append((src_node, tgt_node))
                edge_colors.append(ILLOCUTION_COLORS.get(e["category"], DEFAULT_COLOR))
                edge_widths.append(2.2)
                edge_labels_data[(src_node, tgt_node)] = e["category"]

    # 节点绘制
    nodelist = list(G.nodes())
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodelist,
        node_color=[node_colors.get(n, NODE_COLOR) for n in nodelist],
        edgecolors=NODE_EDGE_COLOR,
        node_size=1800,
        linewidths=1.5,
        ax=ax,
    )

    # 标签只显示 agent 名（去掉 @Rx）
    labels = {n: n.split("@")[0] for n in nodelist}
    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=8, font_weight="bold", font_color="#1A252F",
        ax=ax,
    )

    if edge_list:
        nx.draw_networkx_edges(
            G, pos, edgelist=edge_list,
            edge_color=edge_colors,
            width=edge_widths,
            arrowsize=16,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.1",
            ax=ax,
            node_size=1800,
            alpha=0.85,
        )

        # 简化标签：只显示在边不重叠时
        shown = set()
        for (u, v), lbl in edge_labels_data.items():
            key = (tuple(sorted([u.split("@")[0], v.split("@")[0]])), lbl)
            if key not in shown:
                shown.add(key)
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                ax.text(
                    mid_x, mid_y, lbl,
                    fontsize=6.5, color="#555555", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor="none", alpha=0.8),
                    zorder=5,
                )

    # Y轴标注轮次
    for r in rounds:
        rn = r["round_num"]
        ax.text(
            -0.8, -rn, f"R{rn}",
            fontsize=10, fontweight="bold", color="#555555",
            ha="right", va="center",
        )

    # 图例
    legend_elements = [
        plt.Line2D([0], [0], color=c, lw=3, label=cat)
        for cat, c in ILLOCUTION_COLORS.items()
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        fontsize=9,
        title="Illocutionary Force",
        title_fontsize=10,
        framealpha=0.95,
        edgecolor="#CCCCCC",
    )

    title = "Time-Unfolded Communication DAG"
    if task:
        task_short = task.replace("\n", " ")[:60] + "..." if len(task) > 60 else task
        title += f"\n{task_short}"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color="#1A252F")

    ax.set_xlim(-1.5, len(all_agents) - 0.5)
    ax.set_ylim(-len(rounds) - 0.8, 0.8)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", facecolor="white", dpi=dpi)
    plt.close(fig)


def make_gif(image_paths: list[Path], output_path: Path, duration: int = 1200):
    try:
        from PIL import Image
    except ImportError:
        print("Warning: Pillow not installed, skipping GIF generation.")
        print("  Install with: pip install Pillow")
        return

    images = [Image.open(p) for p in image_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )
    print(f"GIF saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PragmaMAS private_routing communication graphs"
    )
    parser.add_argument("trace_json", help="Path to trace.json")
    parser.add_argument("--output-dir", "-o", default="routing_figures",
                        help="Output directory for figures")
    parser.add_argument("--format", "-f", default="png",
                        choices=["png", "pdf", "svg"], help="Image format")
    parser.add_argument("--gif", action="store_true",
                        help="Generate animated GIF from round figures")
    parser.add_argument("--aggregate", action="store_true",
                        help="Generate aggregate graph (all rounds combined)")
    parser.add_argument("--time-unfolded", action="store_true",
                        help="Generate time-unfolded DAG")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 8],
                        help="Figure size (width height)")
    parser.add_argument("--layout", default="circular",
                        choices=["circular", "spring", "shell", "kamada"],
                        help="Node layout algorithm")
    args = parser.parse_args()

    trace_path = Path(args.trace_json)
    if not trace_path.exists():
        print(f"Error: {trace_path} not found.")
        sys.exit(1)

    data = load_trace(str(trace_path))
    rounds = data.get("rounds", [])
    task = data.get("task", "")
    if not rounds:
        print("Error: No rounds found in trace.")
        sys.exit(1)

    all_agents = build_agent_set(rounds)
    print(f"Agents: {sorted(all_agents)}")
    print(f"Rounds: {len(rounds)}")

    # 统一节点布局（基于全部 Agent）
    G_full = nx.DiGraph()
    G_full.add_nodes_from(sorted(all_agents))
    pos = _make_pos(args.layout, G_full)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 逐轮图 ──
    image_paths = []
    for r in rounds:
        round_num = r["round_num"]
        edges = extract_private_edges(r)
        print(f"Round {round_num}: {len(edges)} private edge(s)")

        out_path = output_dir / f"round_{round_num:03d}.{args.format}"
        draw_round(
            pos, round_num, edges, all_agents,
            out_path, figsize=tuple(args.figsize), dpi=args.dpi, task=task,
        )
        image_paths.append(out_path)
        print(f"  → {out_path}")

    if args.gif and image_paths:
        gif_path = output_dir / "routing_animation.gif"
        make_gif(image_paths, gif_path)

    # ── 聚合图 ──
    if args.aggregate:
        all_edges = []
        for r in rounds:
            all_edges.extend(extract_private_edges(r))
        agg_path = output_dir / f"aggregate.{args.format}"
        draw_aggregate(
            pos, all_edges, all_agents,
            agg_path, figsize=tuple(args.figsize), dpi=args.dpi, task=task,
        )
        print(f"Aggregate → {agg_path}")

    # ── 时间展开图 ──
    if args.time_unfolded:
        tu_path = output_dir / f"time_unfolded.{args.format}"
        draw_time_unfolded(
            rounds, all_agents,
            tu_path, figsize=(14, max(8, len(rounds) * 1.2)), dpi=args.dpi, task=task,
        )
        print(f"Time-unfolded → {tu_path}")

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
