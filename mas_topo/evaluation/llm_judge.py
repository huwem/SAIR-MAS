"""LLM 评判评估器。

适用于 MATH 等数学推理数据集，调用大模型判断预测答案与 ground truth 是否在数学上等价。
比精确匹配更鲁棒，能处理 LaTeX 格式差异、等价变形、分数/根号不同表示等情况。
"""

import json
import logging
import re

from tenacity import RetryError

from mas_topo._registries import evaluator_registry
from mas_topo.datasets.base import DatasetSample
from mas_topo.evaluation.base import Evaluator, EvaluationResult
from mas_topo.evaluation.exact_match import ExactMatchEvaluator
from mas_topo.llm import OpenAICompatAdapter

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a mathematical answer judge. Your task is to determine whether two mathematical answers are equivalent.

Rules:
1. Compare the predicted answer with the ground truth answer.
2. Consider LaTeX formatting differences, equivalent forms (e.g., 1/2 vs 0.5, \\frac{1}{2} vs 0.5), and mathematical equivalence.
3. Ignore minor differences like spacing, variable naming if the mathematical meaning is the same.
4. If the predicted answer is mathematically equivalent to the ground truth, output {"is_correct": true}.
5. If they are different or the prediction is wrong/empty, output {"is_correct": false}.
6. Respond ONLY with a valid JSON object containing a single key "is_correct" with a boolean value.
"""


_JSON_BOOL_RE = re.compile(r'"is_correct"\s*:\s*(true|false)', re.IGNORECASE)


def _parse_llm_judge_response(content: str) -> bool:
    """从 LLM 响应中解析 is_correct 布尔值。

    优先尝试完整 JSON 解析；失败时回退到正则提取，以兼容模型在 JSON mode
    外输出额外文本的情况。
    """
    content = content.strip()
    if not content:
        raise ValueError("LLM judge returned empty content")

    # 1. 直接 JSON 解析
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "is_correct" in parsed:
            return bool(parsed["is_correct"])
    except json.JSONDecodeError:
        pass

    # 2. 回退：从文本中提取 "is_correct": true/false
    match = _JSON_BOOL_RE.search(content)
    if match:
        return match.group(1).lower() == "true"

    raise ValueError(f"Unable to parse LLM judge response: {content[:200]!r}")


def _format_judge_error(exc: Exception) -> str:
    """把 judge 调用异常格式化为可读字符串。

    若 tenacity 重试耗尽，解包最后一次底层异常，避免 details["error"]
    出现 ``RetryError[<Future ...>]`` 这类难以阅读的信息。
    """
    if isinstance(exc, RetryError):
        last_attempt = getattr(exc, "last_attempt", None)
        if last_attempt is not None and getattr(last_attempt, "failed", False):
            last_exc = last_attempt.exception()
            if last_exc is not None:
                return f"{type(last_exc).__name__}: {last_exc}"
    return f"{type(exc).__name__}: {exc}"


@evaluator_registry.register(
    "llm_judge",
    display_name="LLM 评判",
    description="调用大模型判断数学答案是否与 ground truth 等价，支持 LaTeX 格式差异和等价变形。",
    config_schema={
        "model": {"type": "str", "default": "deepseek-v4-flash", "description": "评判所用模型名"},
        "api_key": {"type": "str", "default": "", "description": "API 密钥（空字符串则读环境变量）"},
        "base_url": {"type": "str", "default": "", "description": "API 基础 URL（空字符串则读环境变量）"},
        "temperature": {"type": "float", "default": 0.0, "description": "采样温度"},
        "max_tokens": {"type": "int", "default": 256, "description": "最大生成 token 数"},
    },
)
class LLMJudgeEvaluator(Evaluator):
    """LLM 评判评估器。

    Args:
        model: 评判所用模型名。
        api_key: API 密钥，空字符串则从环境变量读取。
        base_url: API 基础 URL，空字符串则从环境变量读取。
        temperature: 采样温度，建议设为 0 保证确定性。
        max_tokens: 最大生成 token 数。
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.0,
        max_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.metric_name = "llm_judge"
        # 先实例化普通精确匹配评估器，用于快速通过明显正确的样本
        self._exact_matcher = ExactMatchEvaluator(
            extract_boxed=True, normalize_math=True
        )
        self._adapter = OpenAICompatAdapter(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )

    def evaluate(self, sample: DatasetSample, predicted: str) -> EvaluationResult:
        pred = str(predicted) if predicted is not None else ""
        gt = str(sample.ground_truth) if sample.ground_truth is not None else ""

        if not pred.strip():
            return EvaluationResult(
                is_correct=False,
                predicted_answer=pred,
                metric_name=self.metric_name,
                metric_value=0.0,
                details={"error": "empty prediction", "ground_truth": gt},
            )

        # 第一步：普通精确匹配评估
        exact_result = self._exact_matcher.evaluate(sample, predicted)
        if exact_result.is_correct:
            # 精确匹配已正确，无需调用 LLM
            return EvaluationResult(
                is_correct=True,
                predicted_answer=pred,
                metric_name=self.metric_name,
                metric_value=1.0,
                details={
                    "ground_truth": gt,
                    "exact_match": True,
                    "llm_judged": False,
                    "predicted_clean": exact_result.details.get("predicted_clean"),
                    "ground_truth_clean": exact_result.details.get("ground_truth_clean"),
                },
            )

        # 第二步：精确匹配失败，调用 LLM 二次判断
        question = str(sample.question) if sample.question else ""
        question_part = f"Question: {question}\n\n" if question else ""
        user_prompt = f"""{question_part}Ground truth answer: {gt}

Predicted answer: {pred}

Are these two answers mathematically equivalent? Output JSON with key "is_correct".
"""

        llm_response = None
        error_message = None
        try:
            response = self._adapter.call(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]
            )
            content = (response.content or "").strip()
            llm_response = content
            is_correct = _parse_llm_judge_response(content)
        except Exception as e:
            logger.warning("LLM judge failed: %s. Keeping exact_match result.", e)
            is_correct = False
            error_message = _format_judge_error(e)

        return EvaluationResult(
            is_correct=is_correct,
            predicted_answer=pred,
            metric_name=self.metric_name,
            metric_value=1.0 if is_correct else 0.0,
            details={
                "ground_truth": gt,
                "exact_match": False,
                "llm_judged": True,
                "llm_response": llm_response,
                "error": error_message,
            },
        )
