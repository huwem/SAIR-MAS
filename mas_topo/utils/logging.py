"""日志工具。

支持控制台和文件双输出。文件日志按实验目录聚合。
"""

import logging
import os
from pathlib import Path


def setup_logger(name: str = "mas_topo", level: int = logging.INFO) -> logging.Logger:
    """设置并返回一个命名的 logger（仅控制台输出）。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup_file_logging(log_dir: str, level: int = logging.DEBUG) -> str:
    """为根 logger 添加文件 handler，将所有日志同时写入文件。

    Args:
        log_dir: 日志输出目录，日志文件将写入 {log_dir}/experiment.log
        level: 文件日志级别，默认 DEBUG（控制台级别不受影响）

    Returns:
        日志文件路径。
    """
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    log_file = path / "experiment.log"

    root_logger = logging.getLogger()

    # 避免重复添加
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_file):
            return str(log_file)

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info("File logging enabled: %s", log_file)
    return str(log_file)
