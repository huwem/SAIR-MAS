"""代码执行评估器。

适用于 HumanEval 等需要通过执行测试来验证代码正确性的场景。
支持 pass@1 评估，可配置提取代码的方式。
"""

from mas_topo._registries import evaluator_registry
from mas_topo.datasets.base import DatasetSample
from mas_topo.evaluation.base import Evaluator, EvaluationResult
from mas_topo.tools.builtin import extract_python_code, run_python_tests


@evaluator_registry.register(
    "code_execution",
    display_name="代码执行测试",
    description="从预测文本中提取 Python 代码并在隔离子进程中执行测试用例，返回是否通过。",
    config_schema={
        "extract_code": {"type": "bool", "default": True, "description": "是否从 markdown 代码块中提取 Python 代码"},
    },
)
class CodeExecutionEvaluator(Evaluator):
    """代码执行评估器。

    Args:
        extract_code: 若为 True，则从预测文本中提取 ```python 代码块。
    """

    def __init__(self, extract_code: bool = True, **kwargs):
        super().__init__(extract_code=extract_code, **kwargs)
        self.extract_code = extract_code

    def evaluate(self, sample: DatasetSample, predicted: str) -> EvaluationResult:
        pred = str(predicted) if predicted is not None else ""
        test_code = str(sample.ground_truth)

        if not pred.strip():
            return EvaluationResult(
                is_correct=False,
                predicted_answer=pred,
                metric_name="pass@1",
                details={"error": "empty prediction"},
            )

        code = extract_python_code(pred) if self.extract_code else pred
        if not code:
            #  fallback: 如果文本中没有代码块但包含 def/class，直接当代码用
            if "def " in pred or "class " in pred:
                code = pred.strip()
            else:
                return EvaluationResult(
                    is_correct=False,
                    predicted_answer=pred,
                    metric_name="pass@1",
                    details={"error": "no code found"},
                )

        result = run_python_tests(code=code, tests=test_code)
        passed = result.get("passed", False)

        return EvaluationResult(
            is_correct=passed,
            predicted_answer=code,
            metric_value=1.0 if passed else 0.0,
            metric_name="pass@1",
            details={
                "returncode": result.get("returncode"),
                "stdout": result.get("stdout", "")[:500],
                "stderr": result.get("stderr", "")[:500],
            },
        )
