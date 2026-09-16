"""工具模块测试。"""

import subprocess

import pytest

from mas_topo.tools.builtin import (
    MAX_TOOL_TIMEOUT,
    PythonExecutorTool,
    RunPythonTool,
    _clamp_timeout,
    extract_python_code,
)


class TestExtractPythonCode:
    """测试代码提取辅助函数。"""

    def test_extract_from_python_block(self):
        text = 'Some explanation\n```python\ndef f():\n    return 1\n```\nmore'
        assert extract_python_code(text) == "def f():\n    return 1"

    def test_extract_from_plain_block(self):
        text = '```\ndef g():\n    return 2\n```'
        assert extract_python_code(text) == "def g():\n    return 2"

    def test_extract_from_def_lines(self):
        text = "from typing import List\n\ndef h(x: int) -> int:\n    return x + 1"
        assert extract_python_code(text) == text

    def test_extract_assert_after_def(self):
        text = "def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5"
        assert extract_python_code(text) == text

    def test_extract_print_after_def(self):
        text = "def f():\n    return 1\n\nprint(f())"
        assert extract_python_code(text) == text

    def test_extract_assignment_and_assert_after_def(self):
        """def 后紧跟注释、赋值、print、assert 的顶层测试代码应被完整保留。"""
        text = (
            "from typing import List\n\n"
            "def sort_array(arr: List[int]) -> List[int]:\n"
            "    return sorted(arr)\n\n"
            "# Additional check\n"
            "result = sort_array([3, 4])\n"
            "print(result)\n"
            "assert result == [4, 3]"
        )
        assert extract_python_code(text) == text

    def test_empty_input(self):
        assert extract_python_code("") == ""

    def test_extract_plain_stdin_script(self):
        text = "import sys\n\nA, B = map(int, sys.stdin.read().split())\nresult = (A + B) ** 2\nprint(result)"
        assert extract_python_code(text) == text

    def test_extract_plain_input_script(self):
        text = "n = int(input())\nprint(n * 2)"
        assert extract_python_code(text) == text

    def test_extract_preserves_main_block(self):
        text = "def solve():\n    print(1)\n\nif __name__ == '__main__':\n    solve()"
        assert extract_python_code(text) == text

    def test_extract_plain_script_with_if(self):
        text = "n = int(input())\nif n > 0:\n    print(n)\nelse:\n    print(-n)"
        assert extract_python_code(text) == text


class TestRunPythonToolValidation:
    """测试 RunPythonTool 的参数校验逻辑。"""

    @pytest.fixture
    def tool(self):
        return RunPythonTool()

    def test_valid_call_passes(self, tool):
        result = tool.run(
            code="def add(a, b):\n    return a + b\n\nassert add(2, 3) == 5",
        )
        assert result["passed"] is True
        assert result["returncode"] == 0

    def test_valid_call_with_assertions(self, tool):
        result = tool.run(
            code="def identity(x):\n    return x\n\nassert identity(42) == 42\nassert identity('hi') == 'hi'",
        )
        assert result["passed"] is True

    def test_unexpected_argument_key_rejected(self, tool):
        # 传入不在 allowed_keys 中的额外参数
        result = tool.run(
            code="def f():\n    return 1",
            extra_param="unexpected",
        )
        assert result["passed"] is False
        assert "nexpected argument keys" in result["stderr"]
        assert "extra_param" in result["stderr"]

    def test_markdown_code_block_extracted(self, tool):
        result = tool.run(
            code="```python\ndef add(a, b):\n    return a + b\n\nassert add(2, 3) == 5\n```",
        )
        assert result["passed"] is True

    def test_missing_code_rejected(self, tool):
        result = tool.run(code="")
        assert result["passed"] is False
        assert "No code provided" in result["stderr"]

    def test_actual_assertion_failure(self, tool):
        result = tool.run(
            code="def add(a, b):\n    return a + b\n\nassert add(2, 3) == 6",
        )
        assert result["passed"] is False
        assert result["returncode"] != 0

    def test_description_mentions_unified_code(self):
        tool = RunPythonTool()
        desc = tool.description
        assert "BOTH" in desc
        assert "code=" in desc


class TestRunPythonToolNoTruncation:
    """工具参数 code 视为纯代码：无围栏时全文执行，不做行级过滤。

    回归：extract_python_code 策略 4 曾在点号方法调用行（如 sizes.append(...)）
    处静默截断，导致实际执行的残码丢失 print 与核心计算，
    工具返回 PASSED + 空 stdout（sair_reasoning sample_009 事故根因）。
    """

    @pytest.fixture
    def tool(self):
        return RunPythonTool()

    def test_plain_script_with_dotted_calls_runs_in_full(self, tool):
        code = (
            "sizes = []\n"
            "for a in range(2, 5):\n"
            "    for b in range(2, 5):\n"
            "        s = 2 * a + 2 * b - 4\n"
            "        sizes.append((a, b, s))\n"
            "\n"
            "print(f'count={len(sizes)}')\n"
        )
        result = tool.run(code=code)
        assert result["passed"] is True
        assert "count=9" in result["stdout"]

    def test_method_call_lines_not_dropped(self, tool):
        code = (
            "items = set()\n"
            "for x in range(3):\n"
            "    items.add(x * x)\n"
            "print(sorted(items))\n"
        )
        result = tool.run(code=code)
        assert result["passed"] is True
        assert "[0, 1, 4]" in result["stdout"]

    def test_fence_unwrap_still_works(self, tool):
        result = tool.run(code="```python\nprint(6 * 7)\n```")
        assert result["passed"] is True
        assert "42" in result["stdout"]


class TestToolTimeoutClamp:
    """模型传入的 timeout 必须被钳制到 [1, MAX_TOOL_TIMEOUT]。

    回归：schema 允许模型传任意整数 timeout，曾导致 run_python 的失控
    子进程（Fraction 指数爆炸代码）挂起 16h 不死，拖死整个实验
    （sair_reasoning_local aime 20260903_083025, sample_057 事故根因）。
    """

    def test_clamp_huge_value(self):
        assert _clamp_timeout(999999, 30) == MAX_TOOL_TIMEOUT

    def test_clamp_none_falls_back_to_default(self):
        assert _clamp_timeout(None, 30) == 30
        assert _clamp_timeout(None, 5) == 5

    def test_clamp_non_numeric_falls_back_to_default(self):
        assert _clamp_timeout("abc", 30) == 30

    def test_clamp_zero_and_negative_raise_to_one(self):
        assert _clamp_timeout(0, 30) == 1
        assert _clamp_timeout(-10, 30) == 1

    def test_clamp_normal_value_unchanged(self):
        assert _clamp_timeout(10, 30) == 10
        assert _clamp_timeout(MAX_TOOL_TIMEOUT, 30) == MAX_TOOL_TIMEOUT

    def test_run_python_clamps_model_timeout(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("mas_topo.tools.builtin.subprocess.run", fake_run)
        result = RunPythonTool().run(code="print(1)", timeout=999999)
        assert captured["timeout"] == MAX_TOOL_TIMEOUT
        assert result["passed"] is True

    def test_run_python_none_timeout_uses_default(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("mas_topo.tools.builtin.subprocess.run", fake_run)
        RunPythonTool().run(code="print(1)", timeout=None)
        assert captured["timeout"] == 30

    def test_python_executor_clamps_model_timeout(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured.update(kwargs)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr("mas_topo.tools.builtin.subprocess.run", fake_run)
        PythonExecutorTool().run(code="print(1)", timeout=999999)
        assert captured["timeout"] == MAX_TOOL_TIMEOUT

    def test_real_timeout_still_fires(self):
        # 真实子进程：小 timeout 必须按时杀掉长任务（防钳制逻辑弄坏正常路径）
        result = RunPythonTool().run(code="import time\ntime.sleep(10)", timeout=1)
        assert result["passed"] is False
        assert "timed out after 1s" in result["stderr"]


class TestChildMemoryLimit:
    """子进程内存上限：超限应快速 MemoryError，而不是吃光整机内存。

    回归：模型经 run_python 运行失控代码（大额分配/指数增长对象）会把
    整机内存拖爆。执行前注入 RLIMIT_AS 上限（硬限制，模型代码无法绕过）。
    """

    def test_memory_bomb_fails_with_memory_error(self):
        # 分配 5GB：超过上限应立即 MemoryError（修复前会真的吃掉 5GB 并 passed=True）
        result = RunPythonTool().run(code="x = bytearray(5 * 1024**3)")
        assert result["passed"] is False
        assert "MemoryError" in result["stderr"]

    def test_normal_allocation_unaffected(self):
        result = RunPythonTool().run(code="x = bytearray(100 * 1024**2)\nprint(len(x))")
        assert result["passed"] is True
        assert str(100 * 1024**2) in result["stdout"]

    def test_python_executor_memory_bomb_fails(self):
        result = PythonExecutorTool().run(code="x = bytearray(5 * 1024**3)")
        assert "MemoryError" in result["stderr"]

    def test_evaluator_stdin_memory_bomb_fails(self):
        from mas_topo.evaluation.code_execution_stdin import _run_with_stdin

        result = _run_with_stdin("x = bytearray(5 * 1024**3)", "")
        assert result["returncode"] != 0
        assert "MemoryError" in result["stderr"]
