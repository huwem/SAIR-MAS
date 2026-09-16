"""__main__ CLI 入口的参数转发回归测试。

回归：__main__.py 两条内联调用路径显式枚举 kwargs，新增的
--stream/--disable-thinking/--sample-timeout 被静默丢弃，导致
DashScope 思考模式（必须流式）开了 --stream 仍走非流式，全部 400。
"""

import sys

import pytest

import mas_topo.__main__ as main_module


@pytest.fixture
def captured(monkeypatch, tmp_path):
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        "mas_topo.experiment.cli.run_experiment_from_config", fake_run
    )
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("framework:\n  llm:\n    model: x\n")
    return captured, str(cfg)


def test_subcommand_forwards_new_flags(captured):
    captured_kwargs, cfg = captured
    sys_argv = [
        "mas_topo", "experiment", cfg,
        "--stream", "--disable-thinking", "--sample-timeout", "60",
    ]
    sys.argv = sys_argv
    main_module.main()
    assert captured_kwargs["stream"] is True
    assert captured_kwargs["disable_thinking"] is True
    assert captured_kwargs["sample_timeout"] == 60
    assert captured_kwargs["config_path"] == cfg


def test_bare_config_form_forwards_new_flags(captured):
    captured_kwargs, cfg = captured
    sys.argv = [
        "mas_topo", cfg,
        "--stream", "--disable-thinking", "--sample-timeout", "60",
    ]
    main_module.main()
    assert captured_kwargs["stream"] is True
    assert captured_kwargs["disable_thinking"] is True
    assert captured_kwargs["sample_timeout"] == 60
    assert captured_kwargs["config_path"] == cfg
