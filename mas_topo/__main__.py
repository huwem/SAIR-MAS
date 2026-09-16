"""MAS_Topo 包级 CLI 入口。

使用方式:
    # 最简形式：直接传入配置文件（自动识别为 experiment 模式）
    python -m mas_topo configs/my_experiment.yaml
    python -m mas_topo configs/my_experiment.yaml -l 5 -b 3

    # 显式子命令
    python -m mas_topo experiment configs/my_experiment.yaml
    python -m mas_topo run configs/framework.yaml "Task text here"

    # 查看帮助
    python -m mas_topo --help
"""

import argparse
import os
import sys


def _looks_like_config_path(arg: str) -> bool:
    """判断参数是否看起来像配置文件路径。"""
    return arg.endswith((".yaml", ".yml", ".json")) and os.path.exists(arg)


def main() -> None:
    # 检测是否是"最简形式"：第一个位置参数不是子命令，而是配置文件路径
    args = sys.argv[1:]
    if args and _looks_like_config_path(args[0]):
        # 最简形式：直接解析为 experiment 子命令的参数
        from mas_topo.experiment.cli import build_experiment_parser, run_experiment_from_config
        parser = build_experiment_parser()
        parsed = parser.parse_args(args)
        # 直接按 parsed 的命名空间转发，避免新增 CLI 参数被这里遗忘
        kwargs = vars(parsed)
        kwargs["config_path"] = kwargs.pop("config")
        run_experiment_from_config(**kwargs)
        return

    # 显式子命令模式
    parser = argparse.ArgumentParser(
        description="MAS_Topo: Config-driven multi-agent topology experiment framework",
        usage="python -m mas_topo [<config.yaml> | <command>] [options]",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # experiment 子命令
    from mas_topo.experiment.cli import build_experiment_parser
    exp_parser = build_experiment_parser()
    subparsers.add_parser(
        "experiment",
        parents=[exp_parser],
        add_help=False,
        help="Run a dataset experiment from config file",
    )

    # run 子命令
    run_parser = subparsers.add_parser(
        "run",
        help="Run a single task with a framework config",
    )
    run_parser.add_argument("config", type=str, help="Framework config file path")
    run_parser.add_argument("task", type=str, help="Task text")
    run_parser.add_argument(
        "--log-level", type=str, default="warning",
        choices=["debug", "info", "warning", "error", "critical"],
    )

    parsed = parser.parse_args()

    if parsed.command == "experiment":
        from mas_topo.experiment.cli import run_experiment_from_config
        # 直接按 parsed 的命名空间转发，避免新增 CLI 参数被这里遗忘
        kwargs = vars(parsed)
        kwargs.pop("command", None)
        kwargs["config_path"] = kwargs.pop("config")
        run_experiment_from_config(**kwargs)
    elif parsed.command == "run":
        from mas_topo.experiment.cli import run_task_command
        run_task_command()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
