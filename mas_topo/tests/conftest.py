"""mas_topo 单元测试共享配置。

- 为 SentenceTransformer 语义/Cosine 拓扑测试强制使用 CPU，避免 CI/开发机
  GPU 显存不足导致 OOM。
- 提供跨模块共享的通用 fixture。
"""

import os

# 尽量在 PyTorch 初始化前隐藏 GPU 设备
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest


def pytest_configure(config):
    """pytest 初始化时强制语义/Cosine 拓扑的模型加载到 CPU。"""
    try:
        from sentence_transformers import SentenceTransformer
        from mas_topo.topology import semantic
        from mas_topo.topology import cosine
    except Exception:
        return

    def _load_engine_cpu(model_path: str, cache_folder: str | None, local_only: bool):
        return SentenceTransformer(
            model_path,
            cache_folder=cache_folder,
            local_files_only=local_only,
            device="cpu",
        )

    semantic._load_engine = _load_engine_cpu
    cosine._load_engine = _load_engine_cpu


@pytest.fixture
def project_root():
    """返回项目根目录 Path。"""
    from pathlib import Path

    return Path(__file__).resolve().parents[2]
