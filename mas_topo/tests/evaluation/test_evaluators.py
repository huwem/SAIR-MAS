"""评估器单元测试与回归测试。

覆盖 `exact_match` 与 `llm_judge` 两种评估器，重点验证：
- `extract_number=True` 在 MATH 类任务中会导致误判（假阳性/假阴性）。
- `llm_judge` 在精确匹配成功时走快速通道，无需调用 LLM。
- `llm_judge` 在精确匹配失败时调用 LLM 进行二次判断。
"""

from unittest.mock import MagicMock, patch

import pytest
from tenacity import Retrying, stop_after_attempt, wait_fixed

from mas_topo.datasets.math_loader import MATHSample
from mas_topo.evaluation.exact_match import ExactMatchEvaluator
from mas_topo.evaluation.llm_judge import LLMJudgeEvaluator


def _math_sample(ground_truth: str) -> MATHSample:
    """构造一个 ground_truth 为指定值的 MATH 样本。"""
    return MATHSample(
        problem="mock problem",
        solution=f"\\boxed{{{ground_truth}}}",
        level="1",
        problem_type="test",
        index=0,
    )


class TestExactMatchEvaluator:
    """精确匹配评估器回归测试。"""

    @pytest.mark.parametrize(
        ("predicted", "ground_truth"),
        [
            # 含 LaTeX 结构的表达式不做数字提取，避免假阳性
            ("\\frac{23}{3}", "3"),
            ("(5, \\frac{\\pi}{2})", "2"),
        ],
        ids=("fraction", "coordinate"),
    )
    def test_extract_number_false_positives(self, predicted, ground_truth):
        """extract_number=True 不应对 LaTeX 表达式做数字提取（防假阳性）。"""
        evaluator = ExactMatchEvaluator(extract_number=True)
        sample = _math_sample(ground_truth)
        result = evaluator.evaluate(sample, predicted)
        assert result.is_correct is False, (
            f"extract_number=True 不应把 '{predicted}' 与 '{ground_truth}' 判为正确, "
            f"details={result.details}"
        )

    @pytest.mark.parametrize(
        ("predicted", "ground_truth"),
        [
            # 假阴性：相同/等价答案因无数字被清空后无法匹配
            ("Evelyn", "\\text{Evelyn}"),
            ("p - q", "p - q"),
        ],
        ids=("text", "expression"),
    )
    def test_extract_number_false_negatives(self, predicted, ground_truth):
        """extract_number=True 会把非数字答案清空，导致假阴性。"""
        evaluator = ExactMatchEvaluator(extract_number=True)
        sample = _math_sample(ground_truth)
        result = evaluator.evaluate(sample, predicted)
        assert result.is_correct is False, (
            f"extract_number=True 不应把 '{predicted}' 与 '{ground_truth}' 判为正确, "
            f"details={result.details}"
        )

    @pytest.mark.parametrize(
        ("predicted", "ground_truth"),
        [
            ("\\frac{14}{3}", "\\frac{14}{3}"),
            ("(3, \\frac{\\pi}{2})", "\\left( 3, \\frac{\\pi}{2} \\right)"),
            # Unicode 希腊字母应与其 LaTeX/ASCII 等价形式匹配
            ("(3, π/2)", "\\left( 3, \\frac{\\pi}{2} \\right)"),
            ("(3, θ)", "\\left( 3, \\theta \\right)"),
            ("Evelyn", "\\text{Evelyn}"),
            ("p - q", "p - q"),
        ],
        ids=("fraction", "coordinate", "unicode_pi", "unicode_theta", "text", "expression"),
    )
    def test_boxed_normalize_math_matches_complex_answers(self, predicted, ground_truth):
        """不使用 extract_number，仅做 LaTeX 归一化时，复杂答案应正确匹配。"""
        evaluator = ExactMatchEvaluator(extract_boxed=True, normalize_math=True)
        sample = _math_sample(ground_truth)
        result = evaluator.evaluate(sample, predicted)
        assert result.is_correct is True, (
            f"Expected '{predicted}' to match '{ground_truth}', "
            f"details={result.details}"
        )


class TestLLMJudgeEvaluator:
    """LLM 评判评估器测试。"""

    def test_exact_match_short_circuits_no_llm_call(self):
        """精确匹配成功时，LLMJudgeEvaluator 不应发起 LLM 调用。"""
        sample = _math_sample("\\frac{14}{3}")
        evaluator = LLMJudgeEvaluator(model="deepseek-v4-flash")

        with patch("mas_topo.evaluation.llm_judge.OpenAICompatAdapter") as mock_cls:
            result = evaluator.evaluate(sample, "\\frac{14}{3}")
            mock_cls.assert_not_called()

        assert result.is_correct is True
        assert result.details["exact_match"] is True
        assert result.details["llm_judged"] is False

    def test_llm_fallback_when_exact_match_fails(self):
        """精确匹配失败时，LLMJudgeEvaluator 调用 LLM 进行二次判断。"""
        sample = _math_sample("\\frac{14}{3}")

        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '{"is_correct": true}'
        mock_adapter.call.return_value = mock_response

        # 在 patch 范围内实例化 evaluator，确保 self._adapter 是 mock
        with patch(
            "mas_topo.evaluation.llm_judge.OpenAICompatAdapter",
            return_value=mock_adapter,
        ):
            evaluator = LLMJudgeEvaluator(model="deepseek-v4-flash")
            # 数学等价但形式不同，精确匹配会失败，需要 LLM 二次判断
            result = evaluator.evaluate(sample, "\\frac{28}{6}")

        assert result.is_correct is True
        assert result.details["exact_match"] is False
        assert result.details["llm_judged"] is True
        mock_adapter.call.assert_called_once()

    @pytest.mark.parametrize(
        ("raw_response", "expected"),
        [
            ('{"is_correct": true}', True),
            ('{"is_correct": false}', False),
            # JSON mode 外模型可能输出额外解释文本
            ('After comparing, the answers are equivalent.\n{"is_correct": true}', True),
            ('The answers differ. {"is_correct": false}', False),
            # 大小写兼容
            ('{"is_correct": True}', True),
        ],
    )
    def test_llm_response_parsing_robustness(self, raw_response, expected):
        """LLM judge 应能容忍非标准 JSON 输出并正确解析 is_correct。"""
        sample = _math_sample("\\frac{1}{2}")

        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.content = raw_response
        mock_adapter.call.return_value = mock_response

        with patch(
            "mas_topo.evaluation.llm_judge.OpenAICompatAdapter",
            return_value=mock_adapter,
        ):
            evaluator = LLMJudgeEvaluator(model="deepseek-v4-flash")
            result = evaluator.evaluate(sample, "0.5")

        assert result.is_correct is expected
        assert result.details["exact_match"] is False
        assert result.details["llm_judged"] is True
        assert result.details["llm_response"] == raw_response
        assert result.details["error"] is None

    def test_llm_judge_records_error_when_call_fails(self):
        """LLM 调用失败时，details 应记录错误信息而不是只返回 null。"""
        sample = _math_sample("\\frac{1}{2}")

        mock_adapter = MagicMock()
        mock_adapter.call.side_effect = RuntimeError("API unavailable")

        with patch(
            "mas_topo.evaluation.llm_judge.OpenAICompatAdapter",
            return_value=mock_adapter,
        ):
            evaluator = LLMJudgeEvaluator(model="deepseek-v4-flash")
            result = evaluator.evaluate(sample, "0.5")

        assert result.is_correct is False
        assert result.details["exact_match"] is False
        assert result.details["llm_judged"] is True
        assert result.details["llm_response"] is None
        assert "API unavailable" in result.details["error"]

    def test_llm_judge_unwraps_retry_error(self):
        """若 adapter 抛出 tenacity.RetryError，details 应记录底层异常。"""
        sample = _math_sample("\\frac{1}{2}")

        class APIConnectionError(Exception):
            pass

        retry_error = None
        try:
            for attempt in Retrying(wait=wait_fixed(0.01), stop=stop_after_attempt(3)):
                with attempt:
                    raise APIConnectionError("Connection refused by peer")
        except Exception as e:
            retry_error = e

        assert retry_error is not None
        mock_adapter = MagicMock()
        mock_adapter.call.side_effect = retry_error

        with patch(
            "mas_topo.evaluation.llm_judge.OpenAICompatAdapter",
            return_value=mock_adapter,
        ):
            evaluator = LLMJudgeEvaluator(model="deepseek-v4-flash")
            result = evaluator.evaluate(sample, "0.5")

        assert result.is_correct is False
        assert result.details["exact_match"] is False
        assert result.details["llm_judged"] is True
        assert result.details["llm_response"] is None
        assert "RetryError" not in result.details["error"]
        assert "Connection refused by peer" in result.details["error"]
        assert "APIConnectionError" in result.details["error"]
