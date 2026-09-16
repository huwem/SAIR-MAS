"""验证推理类实验配置使用 LLM 评判而非 extract_number 精确匹配。"""

import os

import pytest

from mas_topo.config.loader import _substitute_env_vars, load_config

# 使用 MATH/AIME 数据集的推理类实验配置
REASONING_CONFIG_PATHS = [
    "configs/experiments/dytopo_reasoning_local.yaml",
    "configs/experiments/p2p_free_reasoning_local.yaml",
    "configs/experiments/sair_reasoning_local.yaml",
    "configs/experiments/vanilla_reasoning_local.yaml",
]


@pytest.mark.parametrize("config_path", REASONING_CONFIG_PATHS)
def test_reasoning_config_uses_llm_judge(config_path, project_root):
    """推理任务应使用 llm_judge 评估器，避免 extract_number 导致的误判。"""
    path = project_root / config_path
    if not path.exists():
        pytest.skip(f"Config not found: {config_path}")

    config = load_config(str(path))
    assert config.evaluation.metric == "llm_judge", (
        f"{config_path} 应使用 llm_judge，当前为 {config.evaluation.metric}"
    )
    metric_params = config.evaluation.metric_params
    assert "extract_number" not in metric_params, (
        f"{config_path} 不应再使用 extract_number"
    )


class TestEnvVarSubstitution:
    """环境变量替换行为回归测试。"""

    def test_undefined_env_var_substitutes_to_empty(self):
        """未定义的环境变量应替换为空字符串，而非保留 ${VAR} 占位符。"""
        var_name = "MAS_TOPO_TEST_UNDEFINED_VAR_42"
        os.environ.pop(var_name, None)
        result = _substitute_env_vars(f"${{{var_name}}}")
        assert result == ""

    def test_defined_env_var_substitutes_to_value(self, monkeypatch):
        """已定义的环境变量应替换为实际值。"""
        var_name = "MAS_TOPO_TEST_DEFINED_VAR_42"
        monkeypatch.setenv(var_name, "secret_value")
        result = _substitute_env_vars(f"${{{var_name}}}")
        assert result == "secret_value"

    def test_nested_dict_substitution(self, monkeypatch):
        """嵌套字典中的环境变量应被递归替换。"""
        monkeypatch.setenv("MAS_TOPO_TEST_URL", "https://api.example.com/v1")
        monkeypatch.delenv("MAS_TOPO_TEST_API_KEY", raising=False)
        data = {
            "evaluation": {
                "metric_params": {
                    "api_key": "${MAS_TOPO_TEST_API_KEY}",
                    "base_url": "${MAS_TOPO_TEST_URL}",
                }
            }
        }
        result = _substitute_env_vars(data)
        assert result["evaluation"]["metric_params"]["api_key"] == ""
        assert result["evaluation"]["metric_params"]["base_url"] == "https://api.example.com/v1"
