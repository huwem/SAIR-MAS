"""基于 stdin/stdout 的代码执行评估器。

适用于 LiveCodeBench 等竞赛编程场景：将预测代码作为独立脚本运行，
依次喂入每个测试用例的标准输入，比较标准输出是否与预期一致。
"""

import json
import subprocess

from mas_topo._registries import evaluator_registry
from mas_topo.datasets.base import DatasetSample
from mas_topo.evaluation.base import Evaluator, EvaluationResult
from mas_topo.tools.builtin import extract_python_code, _with_memory_limit


def _run_with_stdin(code: str, stdin_input: str, timeout: int = 30) -> dict:
    """运行 Python 代码并传入 stdin，返回执行结果。"""
    try:
        result = subprocess.run(
            ["python3", "-c", _with_memory_limit(code)],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
        }


@evaluator_registry.register(
    "code_execution_stdin",
    display_name="标准输入输出代码执行测试",
    description="将预测代码作为脚本运行，通过 stdin/stdout 对比验证竞赛编程用例。",
    config_schema={
        "extract_code": {"type": "bool", "default": True, "description": "是否从 markdown 代码块中提取 Python 代码"},
    },
)
class CodeExecutionStdinEvaluator(Evaluator):
    """stdin/stdout 代码执行评估器。

    Args:
        extract_code: 若为 True，则从预测文本中提取 ```python 代码块。
    """

    def __init__(self, extract_code: bool = True, **kwargs):
        super().__init__(extract_code=extract_code, **kwargs)
        self.extract_code = extract_code

    def evaluate(self, sample: DatasetSample, predicted: str) -> EvaluationResult:
        # ground_truth is exposed to agents; private cases are evaluator-only.
        if hasattr(sample, "private_tests"):
            public_tests = list(getattr(sample, "public_tests", []) or [])
            private_tests = list(sample.private_tests or [])
            tests = public_tests + private_tests
            suite = "public+private"
        else:
            public_tests = list(getattr(sample, "public_tests", []) or [])
            private_tests = []
            tests = public_tests
            suite = "public"
            if not tests and sample.ground_truth:
                suite = "ground_truth"
                try:
                    tests = json.loads(sample.ground_truth)
                    if not isinstance(tests, list):
                        tests = [tests]
                except json.JSONDecodeError:
                    tests = []
        details = {
            "suite": suite,
            "n_public_tests": len(public_tests),
            "n_private_tests": len(private_tests),
            "n_tests": len(tests),
            "n_executed": 0,
            "n_passed": 0,
            "test_pass_rate": 0.0,
            "cases": [],
        }
        pred = str(predicted) if predicted is not None else ""
        if not pred.strip():
            return EvaluationResult(
                is_correct=False,
                predicted_answer=pred,
                metric_name="pass@1",
                details={**details, "error": "empty prediction"},
            )

        code = extract_python_code(pred) if self.extract_code else pred
        if not code:
            if "def " in pred or "class " in pred or "input(" in pred or "sys.stdin" in pred:
                code = pred.strip()
            else:
                return EvaluationResult(
                    is_correct=False,
                    predicted_answer=pred,
                    metric_name="pass@1",
                    details={**details, "error": "no code found"},
                )

        if not tests:
            return EvaluationResult(
                is_correct=False,
                predicted_answer=code,
                metric_name="pass@1",
                details={**details, "error": "no test cases found"},
            )

        passed = 0
        for tc in tests:
            stdin_input = tc.get("input", "")
            expected_output = str(tc.get("output", "")).rstrip("\n")
            run_result = _run_with_stdin(code, stdin_input)
            actual_output = run_result["stdout"].rstrip("\n")
            case_passed = (
                run_result["returncode"] == 0
                and actual_output == expected_output
            )
            if case_passed:
                passed += 1
            details["cases"].append({
                "input": stdin_input[:200],
                "expected": expected_output[:200],
                "actual": actual_output[:200],
                "passed": case_passed,
                "stderr": run_result["stderr"][:200],
            })

        all_passed = passed == len(tests)
        details.update(n_executed=len(tests), n_passed=passed,
                       test_pass_rate=passed / len(tests))
        return EvaluationResult(
            is_correct=all_passed,
            predicted_answer=code,
            metric_value=float(all_passed),
            metric_name="pass@1",
            details=details,
        )
