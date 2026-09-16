#!/usr/bin/env python3
"""汇总 MAS_Topo 实验结果为一张 CSV 表。

扫描 experiment_results/ 下所有实验结果目录，按框架方法、数据集、模型
整合为一张 CSV 表，便于对比分析。

用法：
    python scripts/aggregate_experiment_results.py
    python scripts/aggregate_experiment_results.py --input-dir experiment_results --output experiment_results_summary.csv
    python scripts/aggregate_experiment_results.py --input-dir /path/to/experiment_results --output /path/to/summary.csv

参数：
    --input-dir  实验结果根目录，默认 experiment_results
    --output     输出 CSV 路径，默认 experiment_results_summary.csv

输出列：
    framework_method, dataset, model, topology, routing, decision, timestamp,
    results_dir, dataset_path, total_samples, completed_count, correct_count,
    accuracy, metric_name, avg_metric, total_prompt_tokens,
    total_completion_tokens, total_elapsed_time, avg_rounds,
    protocol_completed_count, missing_count
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# 已知策略值（用于从目录名中切分拓扑/路由/决策标签）
TOPOLOGY_TAGS = {"fixed", "semantic", "neighbor"}
ROUTING_TAGS = {"broadcast", "random", "semantic", "self_routing", "tlig_routing", "tlsg_routing", "autogen_groupchat"}
DECISION_TAGS = {"direct", "manager", "consensus", "tlig_consensus", "tlsg_consensus", "autogen_manager"}

# 已知 task 段取值（用于在目录名中定位 task 边界）：
# code/reasoning 为常规实验，math/livecode/humaneval/aime 为 chain 拓扑
# 消融配置（如 sair_math_chain）以数据集名作为 task 段。
TASK_TAGS = {"code", "reasoning", "math", "livecode", "humaneval", "aime"}

# 已知数据集段取值：新版目录名在 scope 段后插入数据集段
# （如 sair_reasoning_local_aime_...），用于区分共享同一配置的不同数据集。
DATASET_TAGS = {"humaneval", "livecodebench", "math", "aime", "gpqa"}

# 已知 scope 段取值（用于确认 task 边界，避免误把方法名中的词当作 task 段）
SCOPE_TAGS = {"local", "chain"}


def _split_strategy_tags(middle: list[str], name: str) -> tuple[str, str, str]:
    """将 middle 部分切分为 topology、routing、decision 三个标签。

    routing_tag 与 decision_tag 可能包含下划线，因此尝试所有可能的切分，
    返回第一个满足三个标签均落在已知策略集合中的组合。
    """
    n = len(middle)
    for di in range(1, 3):
        if di > n - 2:
            continue
        decision_parts = middle[-di:]
        decision = "_".join(decision_parts)
        if decision not in DECISION_TAGS:
            continue
        for ri in range(1, 3):
            if ri > n - di - 1:
                continue
            routing_parts = middle[-di - ri : -di]
            routing = "_".join(routing_parts)
            if routing not in ROUTING_TAGS:
                continue
            topology_parts = middle[: -di - ri]
            topology = "_".join(topology_parts)
            if topology in TOPOLOGY_TAGS:
                return topology, routing, decision
    raise ValueError(f"无法切分策略标签: {'_'.join(middle)} in {name}")


def parse_result_dir_name(name: str) -> dict:
    """从实验结果目录名解析元数据。

    目录名结构（新）:
        {name_tag}_{dataset_tag}_{topology_tag}_{routing_tag}_{decision_tag}_{model_tag}_{timestamp}
    目录名结构（旧，无 dataset_tag）:
        {name_tag}_{topology_tag}_{routing_tag}_{decision_tag}_{model_tag}_{timestamp}
    其中:
        - name_tag = {framework_method}_{task}_{scope}
        - framework_method 可能包含下划线（如 p2p_free、pragmas_p2p、sair_no_irac）
        - task 为 code/reasoning，或数据集名（chain 消融配置，如 math/livecode）
        - scope 为 local 或 chain（chain 消融配置中 chain 落在 scope 段，
          此时 topology_tag 仍为实际的 neighbor 策略）
        - dataset_tag 为数据集名（humaneval/livecodebench/math/aime），旧目录无此段
        - routing_tag / decision_tag 也可能包含下划线（如 self_routing、tlig_consensus）
    """
    parts = name.split("_")
    if len(parts) < 8:
        raise ValueError(f"目录名格式不符: {name}")

    timestamp = "_".join(parts[-2:])
    model = parts[-3]

    # task 边界：取第一个 "task 段 + scope 段" 组合。scope 校验是必须的——
    # 新版目录名中数据集段（math/aime 等）本身也在 TASK_TAGS 中，
    # 不校验 scope 会把数据集段误判为 task 段。
    task_idx = None
    for i in range(len(parts) - 6):
        if parts[i] in TASK_TAGS and parts[i + 1] in SCOPE_TAGS:
            task_idx = i
            break
    if task_idx is None or task_idx == 0:
        raise ValueError(f"目录名中未找到 task 段 {sorted(TASK_TAGS)}: {name}")

    framework_method = "_".join(parts[:task_idx])
    task = parts[task_idx]
    # scope 固定为 task 后的第一段（local 或 chain）
    scope = parts[task_idx + 1]

    # scope 之后可能是数据集段（新格式），其后才是 strategy 三段
    strategy_start = task_idx + 2
    dataset = None
    if parts[strategy_start] in DATASET_TAGS:
        dataset = parts[strategy_start]
        strategy_start += 1

    # 中间部分为 topology_tag + routing_tag + decision_tag
    middle = parts[strategy_start:-3]
    topology, routing, decision = _split_strategy_tags(middle, name)

    return {
        "framework_method": framework_method,
        "task": task,
        "scope": scope,
        "dataset": dataset,
        "topology": topology,
        "routing": routing,
        "decision": decision,
        "model": model,
        "timestamp": timestamp,
    }


def infer_dataset(dataset_path: str | None, task: str) -> str:
    """根据 dataset_path 或 task 推断数据集名称。"""
    if dataset_path:
        path_lower = dataset_path.lower()
        if "humaneval" in path_lower:
            return "humaneval"
        if "livecodebench" in path_lower or "test6" in path_lower:
            return "livecodebench"
        if "aime" in path_lower:
            return "aime"
        if "math" in path_lower:
            return "math"
    if task == "code":
        return "humaneval"
    if task == "reasoning":
        return "math"
    # chain 消融配置以数据集名作为 task 段，直接映射
    dataset_by_task = {
        "math": "math",
        "livecode": "livecodebench",
        "humaneval": "humaneval",
        "aime": "aime",
    }
    if task in dataset_by_task:
        return dataset_by_task[task]
    raise ValueError(f"无法推断数据集: path={dataset_path}, task={task}")


def load_record(results_dir: Path) -> dict | None:
    """从单个实验结果目录读取一条汇总记录。"""
    # 只提取这些汇总字段；results.json 顶层可能还有 subset_note 等
    # 合并来源注释，不属于汇总列，必须忽略（否则写 CSV 时列不一致会报错）
    SUMMARY_KEYS = (
        "dataset_path", "total_samples", "completed_count", "correct_count",
        "accuracy", "metric_name", "avg_metric", "total_prompt_tokens",
        "total_completion_tokens", "total_elapsed_time", "avg_rounds",
    )
    results_file = results_dir / "results.json"
    if not results_file.exists():
        print(f"[SKIP] 无 results.json: {results_dir.name}", file=sys.stderr)
        return None
    try:
        data = json.loads(results_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[SKIP] JSON 解析失败: {results_dir.name}: {e}", file=sys.stderr)
        return None
    # results.json 可能把 summary 直接作为顶层字段，也可能嵌套在 summary 中
    if isinstance(data.get("summary"), dict):
        summary = data["summary"]
    elif isinstance(data.get("results"), list):
        # 顶层字段即为 summary，排除样本级 results 列表
        summary = {k: v for k, v in data.items() if k != "results"}
    else:
        print(f"[SKIP] 无 summary 字段: {results_dir.name}", file=sys.stderr)
        return None
    if not summary:
        print(f"[SKIP] summary 为空: {results_dir.name}", file=sys.stderr)
        return None
    try:
        meta = parse_result_dir_name(results_dir.name)
    except ValueError as e:
        print(f"[SKIP] 目录名解析失败: {results_dir.name}: {e}", file=sys.stderr)
        return None
    # 新格式目录名自带数据集段，优先使用；旧格式回退到按 dataset_path/task 推断
    dataset = meta["dataset"] or infer_dataset(summary.get("dataset_path"), meta["task"])
    record = {
        "framework_method": meta["framework_method"],
        "scope": meta["scope"],
        "dataset": dataset,
        "model": meta["model"],
        # chain 消融目录名中的 topology_tag 是实际路由用的 neighbor 策略，
        # 汇总表统一记为 chain（拓扑受限实验的对外口径）
        "topology": "chain" if meta["scope"] == "chain" else meta["topology"],
        "routing": meta["routing"],
        "decision": meta["decision"],
        "timestamp": meta["timestamp"],
        "results_dir": results_dir.name,
    }
    record.update({k: summary[k] for k in SUMMARY_KEYS if k in summary})

    # ``completed_count`` in results.json is decision-strategy specific:
    # some strategies mark a max-round safety stop as incomplete, while others
    # mark it complete.  The aggregate table uses sample coverage instead:
    # only absent/out-of-scope sample indices are incomplete.  Preserve the
    # original protocol count separately for diagnostics and supplementary
    # analysis.
    result_items = data.get("results", []) if isinstance(data.get("results"), list) else []
    total = int(record.get("total_samples") or 0)
    if dataset == "livecodebench":
        expected_indices = set(range(min(total, 112)))
    else:
        expected_indices = set(range(total))
    observed_indices = {
        item.get("index") for item in result_items
        if isinstance(item, dict) and item.get("index") in expected_indices
    }
    record["protocol_completed_count"] = int(summary.get("completed_count") or 0)
    record["completed_count"] = len(observed_indices) if result_items else record.get("completed_count", 0)
    record["missing_count"] = max(0, total - record["completed_count"])
    return record


def deduplicate_records(records: list[dict]) -> list[dict]:
    """按 (framework_method, scope, dataset, model) 去重。

    完整样本覆盖优先于时间戳；只有覆盖率相同时才保留最新运行。

    scope 必须在键中：chain 消融目录（如 sair_math_chain_...）解析出
    framework_method=sair、scope=chain，与主实验（scope=local）同方法同
    数据集，不带 scope 会被误并为一条。
    """
    groups: dict[tuple, dict] = {}
    for record in records:
        key = (record["framework_method"], record.get("scope"), record["dataset"], record["model"])
        existing = groups.get(key)
        def quality(item: dict) -> tuple[int, int, int, str]:
            total = int(item.get("total_samples") or 0)
            covered = int(item.get("completed_count") or 0)
            # A failed provider run can still emit one placeholder record per
            # sample (often max_rounds=10) and therefore look fully covered.
            # Treat runs with no model tokens as invalid before timestamp-based
            # recency selection; otherwise they overwrite valid checkpoints.
            prompt_tokens = int(item.get("total_prompt_tokens") or 0)
            completion_tokens = int(item.get("total_completion_tokens") or 0)
            has_model_output = int(prompt_tokens + completion_tokens > 0)
            return (
                int(total > 0 and covered >= total),
                has_model_output,
                covered,
                item.get("timestamp", ""),
            )
        if existing is None or quality(record) > quality(existing):
            groups[key] = record
    return list(groups.values())


def write_csv(records: list[dict], output_path: Path) -> None:
    """将记录写入 CSV 文件。"""
    if not records:
        print("[WARN] 无记录可输出", file=sys.stderr)
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_metric_views(
    records: list[dict], accuracy_path: Path, pass_at_1_path: Path
) -> None:
    """Write normalized percentage views derived from the canonical summary."""
    common_fields = [
        "framework_method", "scope", "dataset", "model", "topology",
        "routing", "decision", "accuracy_percent", "correct_count",
        "total_samples", "coverage_percent", "avg_metric_percent",
        "missing_count", "metric_name", "timestamp", "results_dir",
    ]
    rows = []
    for record in records:
        total = int(record.get("total_samples") or 0)
        completed = int(record.get("completed_count") or 0)
        rows.append({
            "framework_method": record.get("framework_method", ""),
            "scope": record.get("scope", ""),
            "dataset": record.get("dataset", ""),
            "model": str(record.get("model", "")).split(":", 1)[0],
            "topology": record.get("topology", ""),
            "routing": record.get("routing", ""),
            "decision": record.get("decision", ""),
            "accuracy_percent": round(float(record.get("accuracy") or 0) * 100, 1),
            "correct_count": int(record.get("correct_count") or 0),
            "total_samples": total,
            "coverage_percent": round(completed / total * 100, 1) if total else 0.0,
            "avg_metric_percent": round(float(record.get("avg_metric") or 0) * 100, 1),
            "missing_count": int(record.get("missing_count") or 0),
            "metric_name": record.get("metric_name", ""),
            "timestamp": record.get("timestamp", ""),
            "results_dir": record.get("results_dir", ""),
        })

    with accuracy_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=common_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    pass_fields = [field for field in common_fields if field != "metric_name"]
    with pass_at_1_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pass_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            if row["metric_name"] == "pass@1":
                writer.writerow({field: row[field] for field in pass_fields})


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总 MAS_Topo 实验结果为一张 CSV 表")
    parser.add_argument("--input-dir", default="experiment_results", help="实验结果根目录")
    parser.add_argument("--output", default="experiment_results_summary.csv", help="输出 CSV 路径")
    parser.add_argument(
        "--accuracy-output", default="experiment_results_accuracy.csv",
        help="整题正确率百分比视图",
    )
    parser.add_argument(
        "--pass-at-1-output", default="experiment_results_pass_at_1.csv",
        help="仅 pass@1 运行的百分比视图",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_dir}", file=sys.stderr)
        return 1

    records = []
    for subdir in sorted(input_dir.iterdir()):
        if not subdir.is_dir():
            continue
        record = load_record(subdir)
        if record is not None:
            records.append(record)

    records = deduplicate_records(records)
    write_csv(records, output_path)
    write_metric_views(
        records, Path(args.accuracy_output), Path(args.pass_at_1_output)
    )
    print(f"[OK] 已生成汇总表: {output_path} (共 {len(records)} 条记录)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
