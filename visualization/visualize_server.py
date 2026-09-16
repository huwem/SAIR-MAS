"""PragmaMAS 交互式实验可视化 Web 服务。

用法:
    python visualization/visualize_server.py --experiment-dir experiment_results --port 8333
"""

import argparse
import json
from pathlib import Path

from flask import Flask, jsonify, render_template


def create_app(experiment_dir: str) -> Flask:
    """创建 Flask 应用。"""
    app = Flask(__name__, template_folder="templates", static_folder="static")
    exp_path = Path(experiment_dir).resolve()

    @app.route("/")
    def index():
        return render_template("visualize.html")

    @app.route("/api/runs")
    def api_runs():
        runs = []
        if exp_path.exists():
            for p in exp_path.iterdir():
                if p.is_dir() and (p / "trace").is_dir():
                    runs.append(p.name)
        return jsonify({"runs": sorted(runs)})

    @app.route("/api/runs/<run>/samples")
    def api_samples(run: str):
        run_path = exp_path / run / "trace"
        samples = []
        if run_path.exists():
            for p in run_path.iterdir():
                if p.is_dir() and (p / "trace.json").exists():
                    samples.append(p.name)
        return jsonify({"samples": sorted(samples)})

    @app.route("/api/runs/<run>/samples/<sample>")
    def api_trace(run: str, sample: str):
        trace_path = exp_path / run / "trace" / sample / "trace.json"
        if not trace_path.exists():
            return jsonify({"error": "Trace not found"}), 404
        with open(trace_path, "r", encoding="utf-8") as f:
            trace = json.load(f)
        # ledger 可能在 trace 顶层或各轮次中
        ledger_entries = trace.get("ledger", [])
        return jsonify({
            "trace": trace,
            "round_tree": build_round_tree(trace),
            "ledger": ledger_entries,
        })

    return app


def build_round_tree(trace: dict) -> dict:
    """按轮次构建基于通信流的层级树。

    节点：每个 (agent, round)。
    边：对 round r 的每条消息 source->target，添加 source@Rr -> target@R(r+1)
        （消息在 round r 发送，在 round r+1 被接收和处理）。
    为避免层级断裂，每一轮的所有 agent 都创建节点（即使该轮未收发消息）。
    """
    nodes = []
    edges = []
    node_ids = set()
    rounds = trace.get("rounds", [])

    def ensure_node(agent: str, rn: int):
        node_id = f"{agent}@R{rn}"
        if node_id not in node_ids:
            node_ids.add(node_id)
            nodes.append({
                "id": node_id,
                "label": f"{agent}\nR{rn}",
                "level": rn,
                "agent": agent,
                "round": rn,
            })
        return node_id

    # 从所有轮次中提取全部 agent 名称
    all_agents = set()
    for r in rounds:
        for name in r.get("agents", {}).keys():
            all_agents.add(name)
        for name in r.get("adjacency_list", {}).keys():
            all_agents.add(name)
    max_round = max((r.get("round_num", 0) for r in rounds), default=0)

    # 为每一轮 + 最后一轮消息的接收层预创建所有 agent 节点
    # 消息 source@Rr -> target@R(r+1)，因此需要 max_round+1 层
    if all_agents:
        for rn in range(1, max_round + 2):
            for agent in sorted(all_agents):
                ensure_node(agent, rn)

    for round_data in rounds:
        rn = round_data.get("round_num", 0)
        for msg in round_data.get("edges", []):
            src = msg.get("source", "?")
            tgt = msg.get("target", "?")
            # 消息在 round r 发送，在 round r+1 被接收
            src_id = ensure_node(src, rn)
            tgt_id = ensure_node(tgt, rn + 1)
            edges.append({
                "from": src_id,
                "to": tgt_id,
                "content": msg.get("content", "")[:500],
                "channel": msg.get("channel", ""),
                "illocution": msg.get("metadata", {}).get("illocution_category", ""),
            })

    # 为每个 agent 分配固定列位置，使同一 agent 在每轮垂直对齐
    sorted_agents = sorted(all_agents)
    agent_to_x = {agent: i * 220 for i, agent in enumerate(sorted_agents)}
    for node in nodes:
        if "agent" in node:
            node["x"] = agent_to_x.get(node["agent"], 0)
            node["y"] = node["level"] * 180  # level 越大越靠下

    return {"nodes": nodes, "edges": edges}


def main():
    parser = argparse.ArgumentParser(description="PragmaMAS experiment visualizer")
    parser.add_argument("--experiment-dir", default="experiment_results",
                        help="Directory containing experiment results")
    parser.add_argument("--port", type=int, default=8333,
                        help="Port to run the server on")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Host to bind to")
    args = parser.parse_args()

    app = create_app(args.experiment_dir)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
