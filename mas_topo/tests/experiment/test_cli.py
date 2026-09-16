"""实验 CLI 单元测试。"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mas_topo.config.loader import load_config
from mas_topo.experiment.cli import DATASET_PRESETS, run_experiment_from_config


MINIMAL_CONFIG = """
name: test_experiment
dataset:
  loader: humaneval
  path: datasets/humaneval/humaneval-py.jsonl
  limit: 1
  offset: 0
evaluation:
  metric: code_execution
  metric_params:
    extract_code: true
output:
  dir: null
  formats: [json]
framework:
  llm:
    provider: openai_compat
    model: deepseek-v4-flash
    api_key: ${OPENAI_API_KEY}
    base_url: ${OPENAI_BASE_URL}
    temperature: 0.3
    max_tokens: 1000
    json_mode: false
  agents:
    - type: LLMAgent
      name: Developer
      role: Coder
      output_format: text
      required_output_fields: [answer]
  topology:
    strategy: neighbor
  routing:
    strategy: broadcast
  decision:
    strategy: direct
  memory:
    strategy: default
  max_rounds: 1
  verbose: false
"""


@pytest.fixture
def minimal_config_path(tmp_path: Path, monkeypatch):
    """创建最小可加载实验配置文件，并固定输出目录。"""
    config_path = tmp_path / "minimal.yaml"
    config_path.write_text(MINIMAL_CONFIG, encoding="utf-8")
    # 避免测试产生 experiment_results 目录
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9999")
    return str(config_path)


class TestExperimentCLI:
    """实验 CLI 行为测试。"""

    def test_dataset_preset_metric_params_substitute_env_vars(
        self, minimal_config_path, tmp_path, monkeypatch
    ):
        """--dataset 使用 preset 时，metric_params 中的 ${...} 应被环境变量替换。"""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://env.example.com")

        captured_evaluator_kwargs = {}

        def fake_evaluator_get(key, **kwargs):
            captured_evaluator_kwargs[key] = kwargs
            return MagicMock()

        def fake_loader_get(key, **kwargs):
            loader = MagicMock()
            loader.total_count.return_value = 1
            return loader

        with patch("mas_topo.experiment.cli.evaluator_registry.get", fake_evaluator_get):
            with patch("mas_topo.experiment.cli.dataset_registry.get", fake_loader_get):
                with patch("mas_topo.experiment.cli.DatasetExperimentRunner") as mock_runner_cls:
                    mock_runner = MagicMock()
                    mock_runner.run_on_dataset.return_value = {}
                    mock_runner_cls.return_value = mock_runner
                    run_experiment_from_config(
                        config_path=minimal_config_path,
                        dataset="aime",
                        output_dir=str(tmp_path / "out"),
                    )

        assert captured_evaluator_kwargs["llm_judge"]["api_key"] == "sk-from-env"
        assert captured_evaluator_kwargs["llm_judge"]["base_url"] == "http://env.example.com"
        assert captured_evaluator_kwargs["llm_judge"]["model"] == "deepseek-v4-flash"

    def test_dataset_preset_metric_params_fall_back_to_framework_llm(
        self, minimal_config_path, tmp_path, monkeypatch
    ):
        """当环境变量未设置时，preset 的 api_key/base_url 应允许被 framework.llm 覆盖。"""
        # 清除环境变量，让 _substitute_env_vars 返回空字符串
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        captured_evaluator_kwargs = {}

        def fake_evaluator_get(key, **kwargs):
            captured_evaluator_kwargs[key] = kwargs
            return MagicMock()

        def fake_loader_get(key, **kwargs):
            loader = MagicMock()
            loader.total_count.return_value = 1
            return loader

        with patch("mas_topo.experiment.cli.evaluator_registry.get", fake_evaluator_get):
            with patch("mas_topo.experiment.cli.dataset_registry.get", fake_loader_get):
                with patch("mas_topo.experiment.cli.DatasetExperimentRunner") as mock_runner_cls:
                    mock_runner = MagicMock()
                    mock_runner.run_on_dataset.return_value = {}
                    mock_runner_cls.return_value = mock_runner
                    # 通过命令行传入 framework.llm 的 key/base_url
                    run_experiment_from_config(
                        config_path=minimal_config_path,
                        dataset="aime",
                        api_key="sk-from-cli",
                        base_url="http://cli.example.com",
                        output_dir=str(tmp_path / "out"),
                    )

        assert captured_evaluator_kwargs["llm_judge"]["api_key"] == "sk-from-cli"
        assert captured_evaluator_kwargs["llm_judge"]["base_url"] == "http://cli.example.com"

    def test_dataset_presets_contain_unresolved_env_placeholders(self):
        """DATASET_PRESETS 自身保留 ${...} 占位符，由运行时替换。"""
        params = DATASET_PRESETS["aime"]["metric_params"]
        assert params["api_key"] == "${OPENAI_API_KEY}"
        assert params["base_url"] == "${OPENAI_BASE_URL}"


class TestDisableThinkingFlag:
    """--disable-thinking 命令行开关。

    DashScope 系网关节点对非流式调用强制要求 enable_thinking=false，
    qwen3 换网关时需要按 run 打开，不宜写死进共享 yaml。
    """

    def test_parser_accepts_disable_thinking_flag(self):
        from mas_topo.experiment.cli import build_experiment_parser

        parser = build_experiment_parser()
        assert parser.parse_args(["cfg.yaml", "--disable-thinking"]).disable_thinking is True
        assert parser.parse_args(["cfg.yaml"]).disable_thinking is False

    def test_parser_accepts_stream_flag(self):
        from mas_topo.experiment.cli import build_experiment_parser

        parser = build_experiment_parser()
        assert parser.parse_args(["cfg.yaml", "--stream"]).stream is True
        assert parser.parse_args(["cfg.yaml"]).stream is False
