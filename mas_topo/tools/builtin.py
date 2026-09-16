"""内置工具注册。

提供常用工具的注册实现。
"""

import os
import re
import subprocess
from typing import Any, Optional

from mas_topo._registries import tool_registry
from mas_topo.tools.base import Tool

# run_python 工具的统一名称；框架内针对该工具的特判逻辑（结果摘要、
# vote_complete 诚实性约束、路由验证状态提取）均引用此常量。
RUN_PYTHON_TOOL_NAME = "run_python"

# run_python 工具的统一名称；框架内针对该工具的特判逻辑（结果摘要、
# vote_complete 诚实性约束、路由状态提取）均引用此常量。
RUN_PYTHON_TOOL_NAME = "run_python"

# 模型经工具参数传入的 timeout 上限（秒）。schema 允许模型给出任意整数，
# 极大值会让失控子进程长时间阻塞实验（曾有 Fraction 爆炸代码跑 16h 不死）。
MAX_TOOL_TIMEOUT = 120

# 子进程地址空间上限（字节）。模型生成的代码可能失控分配内存（大额
# bytearray、指数增长的 Fraction 等），把整机拖垮；执行前注入 RLIMIT_AS。
MAX_CHILD_MEMORY_BYTES = 4 * 1024**3

# 注入到待执行代码前的资源上限前导。硬限制 == 软限制，模型代码无法自行调高。
_LIMITS_PREAMBLE = (
    "import resource\n"
    f"resource.setrlimit(resource.RLIMIT_AS, ({MAX_CHILD_MEMORY_BYTES}, {MAX_CHILD_MEMORY_BYTES}))\n"
)


def _with_memory_limit(code: str) -> str:
    """在执行代码前注入内存上限（RLIMIT_AS）。会让 traceback 行号偏移 2 行。"""
    return _LIMITS_PREAMBLE + code


def _clamp_timeout(timeout: Any, default: int) -> int:
    """将模型传入的 timeout 收敛到 [1, MAX_TOOL_TIMEOUT]，非法值回退默认值。"""
    try:
        t = int(timeout)
    except (TypeError, ValueError):
        return default
    return max(1, min(t, MAX_TOOL_TIMEOUT))


# ── 代码提取工具函数 ──────────────────────────────────────────

def extract_python_code(text: str) -> str:
    """从 LLM 输出文本中提取 Python 函数/类代码。

    供 HumanEvalSample、代码执行评估等"从自由文本中抠代码"的模块复用。
    RunPythonTool 不使用本函数——工具参数 code 视为纯代码，
    仅经 _unwrap_code_fences 剥离围栏，不做行级过滤。

    优先级：
    1. ```python ... ``` 代码块
    2. ``` ... ``` 代码块
    3. 以 def / class 开头的连续缩进行（含前置 import）
    """
    if not text:
        return ""

    # 策略 1: ```python 代码块
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 策略 2: ``` 代码块
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 策略 3: def / class 开头的连续行，回溯包含前置 import 语句
    lines = text.split("\n")
    code_lines = []
    in_block = False
    def_start_idx = -1
    last_collected_idx = -1

    def _looks_like_code(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if (
            stripped.startswith("assert ")
            or stripped.startswith("print(")
            or stripped.startswith("#")
            or stripped.startswith("from ")
            or stripped.startswith("import ")
            or stripped.startswith("@")
            or stripped.startswith("if __name__")
            or bool(re.match(r"^\w+\(", stripped))
            # 赋值语句（含元组解包、增量赋值），排除 == 比较
            or bool(re.match(r"^[A-Za-z_][\w\.\[\]\"', ]*[\+\-\*/%&\|\^]?=(?!=)", stripped))
        ):
            return True
        # 顶层控制流语句（LiveCodeBench 纯脚本常见）
        if re.match(r"^(if|elif|for|while|try|with)\b.*:$", stripped):
            return True
        if re.match(r"^(else|finally|except)\b.*:$", stripped):
            return True
        return False

    for i, line in enumerate(lines):
        if line.startswith("def ") or line.startswith("class "):
            if def_start_idx < 0:
                def_start_idx = i
            code_lines.append(line)
            in_block = True
            last_collected_idx = i
        elif in_block and (line.startswith("    ") or line.startswith("\t")):
            code_lines.append(line)
            last_collected_idx = i
        elif in_block and line.strip() == "":
            code_lines.append(line)
            last_collected_idx = i
        elif in_block and not line.startswith(" "):
            # def/class 缩进块结束，检查是否后续的顶层代码语句
            if _looks_like_code(line):
                code_lines.append(line)
                last_collected_idx = i
            else:
                break

    # 继续收集 def/class 之后的额外顶层代码行（assert / 函数调用等）
    if in_block and def_start_idx >= 0 and last_collected_idx >= 0:
        for i in range(last_collected_idx + 1, len(lines)):
            line = lines[i]
            if _looks_like_code(line) or line.strip() == "":
                code_lines.append(line)
            else:
                break

    if code_lines and def_start_idx >= 0:
        # 回溯：包含 def 之前的 import / from 语句、常量/变量定义与注释
        preamble = []
        for i in range(0, def_start_idx):
            stripped = lines[i].strip()
            if stripped == "":
                continue
            if (
                stripped.startswith("import ")
                or stripped.startswith("from ")
                or stripped.startswith("#")
                or stripped.startswith("@")
                or re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", stripped)
            ):
                preamble.append(lines[i])
            # 非代码行（如自然语言说明）直接跳过，不中断扫描
        if preamble:
            code_lines = preamble + [""] + code_lines
        return "\n".join(code_lines).strip()

    # 策略 4: 无 def/class 的纯 stdin/stdout 脚本（LiveCodeBench 常见）
    stripped = text.strip()
    if stripped and (
        "input(" in stripped
        or "sys.stdin" in stripped
        or re.search(r"^\s*(import |from )", stripped, re.MULTILINE)
        or re.search(r"^\s*print\(", stripped, re.MULTILINE)
    ):
        # 取从第一个看起来像代码的行开始到第一个非代码行的连续块
        code_lines = []
        started = False
        for line in lines:
            s = line.strip()
            if not started:
                if (
                    s.startswith("import ")
                    or s.startswith("from ")
                    or s.startswith("#")
                    or s.startswith("print(")
                    or "=" in s
                    or "input(" in s
                    or "sys.stdin" in s
                    or _looks_like_code(line)
                ):
                    started = True
                else:
                    continue
            if started:
                if s == "" or _looks_like_code(line) or s.startswith("print(") or "=" in s or "input(" in s or "sys.stdin" in s:
                    code_lines.append(line)
                else:
                    break
        if code_lines:
            return "\n".join(code_lines).strip()

    return ""


def _unwrap_code_fences(text: str) -> str:
    """剥离 markdown 代码围栏；无围栏时原文返回。

    专供 RunPythonTool：工具参数的 code 本就是纯代码，不做行级过滤
    （extract_python_code 的策略 3/4 会在点号方法调用等处静默截断，
    曾导致工具实际执行的代码丢失 print 与核心计算）。
    """
    if not text:
        return ""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_python_tests(code: str, tests: str, timeout: int = 30) -> dict:
    """将实现代码与测试用例合并后执行，返回 run_python 工具的结果。

    这是独立的“代码验证”辅助函数，与 Agent 调用的 run_python 工具分离：
    工具只负责执行合并后的代码，而本函数负责把实现和测试用例拼成一份可执行代码。

    Args:
        code: 实现代码（调用前应已去除 markdown 代码块等包装）。
        tests: 测试用例代码。
        timeout: 执行超时秒数。

    Returns:
        {"passed": bool, "stdout": str, "stderr": str, "returncode": int}
    """
    clean_code = code.strip()
    clean_tests = tests.strip()
    if not clean_code:
        return {"passed": False, "stdout": "", "stderr": "No code provided.", "returncode": -1}

    combined = f"{clean_code.rstrip()}\n\n{clean_tests}"
    return RunPythonTool().run(code=combined, timeout=timeout)


# ── 内置工具 ──────────────────────────────────────────────────

@tool_registry.register("python_executor")
class PythonExecutorTool(Tool):
    """Python 代码执行器工具。"""

    @property
    def name(self) -> str:
        return "python_executor"

    @property
    def description(self) -> str:
        return "Execute Python code and return the result."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute."},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 5)."},
            },
            "required": ["code"],
        }

    def run(self, code: str = "", timeout: int = 5, **kwargs) -> Any:
        timeout = _clamp_timeout(timeout, 5)
        try:
            result = subprocess.run(
                ["python", "-c", _with_memory_limit(code)],
                capture_output=True, text=True, timeout=timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Execution timed out after {timeout}s"}


@tool_registry.register("python_test")  # 兼容旧配置中的工具名
@tool_registry.register(RUN_PYTHON_TOOL_NAME)
class RunPythonTool(Tool):
    """Python 代码测试工具。

    将实现代码与测试用例组合后在隔离子进程中执行，
    返回通过/失败状态及执行输出。

    使用场景：
    - Agent 在推理过程中自测生成的代码
    - HumanEval 等基准的 pass@1 评估
    - 通用代码验证流水线
    """

    @property
    def name(self) -> str:
        return RUN_PYTHON_TOOL_NAME

    @property
    def description(self) -> str:
        return (
            "Execute Python code in an isolated subprocess and return the result. "
            "Put BOTH the function implementation AND test assertions together in 'code'. "
            "Example: code='def add(a, b):\\n    return a + b\\n\\nassert add(2, 3) == 5' "
            "Returns {passed: bool, stdout: str, stderr: str, returncode: int}."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Complete Python code to execute, including both the function "
                        "implementation AND test assertions in a single string. "
                        "May contain markdown code blocks (auto-extracted)."
                    ),
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (default: 30).",
                },
            },
            "required": ["code"],
        }

    @staticmethod
    def _validate_arguments(arguments: dict) -> tuple[bool, str]:
        """校验 run_python 工具参数格式。"""
        allowed_keys = {"code", "timeout"}
        unexpected = set(arguments.keys()) - allowed_keys
        if unexpected:
            return False, (
                f"Unexpected argument keys {sorted(unexpected)}. "
                f"Allowed keys are {sorted(allowed_keys)}. "
                "Put everything (implementation + tests) in the 'code' argument."
            )

        code = arguments.get("code", "")
        if not code or not code.strip():
            return False, "No code provided."

        return True, ""

    def run(
        self,
        code: str = "",
        timeout: int = 30,
        **kwargs,
    ) -> dict:
        """在子进程中执行 Python 代码。

        Args:
            code: 完整 Python 代码（实现 + 测试断言，可含 markdown 代码块自动提取）。
            timeout: 执行超时秒数。

        Returns:
            {"passed": bool, "stdout": str, "stderr": str, "returncode": int}
        """
        # 过滤框架注入的内部参数（如 agent）
        timeout = _clamp_timeout(timeout, 30)
        user_kwargs = {k: v for k, v in kwargs.items() if k != "agent"}
        arguments = {"code": code, "timeout": timeout, **user_kwargs}
        ok, err_msg = self._validate_arguments(arguments)
        if not ok:
            return {"passed": False, "stdout": "", "stderr": err_msg, "returncode": -1}

        # 工具参数 code 视为纯代码：仅剥离 markdown 围栏，不做行级过滤，
        # 绝不截断（截断会静默丢失模型代码的后半段，见 _unwrap_code_fences）
        clean_code = _unwrap_code_fences(code)
        if not clean_code:
            return {"passed": False, "stdout": "", "stderr": "No executable code found in input", "returncode": -1}

        try:
            result = subprocess.run(
                ["python3", "-c", _with_memory_limit(clean_code)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return {
                "passed": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"passed": False, "stdout": "", "stderr": f"Execution timed out after {timeout}s", "returncode": -1}
        except Exception as e:
            return {"passed": False, "stdout": "", "stderr": str(e), "returncode": -1}


@tool_registry.register("echo")
class EchoTool(Tool):
    """回显工具（用于测试）。"""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo back the input text."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo back."},
            },
            "required": ["text"],
        }

    def run(self, text: str = "", **kwargs) -> str:
        return text


@tool_registry.register("update_topology_view")
class UpdateTopologyViewTool(Tool):
    """更新 Agent 统一局部邻居表的工具。

    Agent 通过调用该工具，将学习到的其他 Agent 的邻居关系写入自己的
    local_neighbor_table。该工具只更新 Agent 的主观认知条目（非自身 key），
    不改变由框架维护的自身物理一跳邻居条目，也不改变物理拓扑。
    """

    @property
    def name(self) -> str:
        return "update_topology_view"

    @property
    def description(self) -> str:
        return (
            "Update your local neighbor table based on learned neighbor information. "
            "This only updates your cognitive entries for OTHER agents and does NOT change physical topology. "
            "Returns: {success: bool, updated: {agent_name: [neighbors]}}."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "topology_info": {
                    "type": "object",
                    "description": (
                        "Dictionary mapping agent names to lists of their direct neighbors, "
                        "e.g. {'Agent2': ['Agent1', 'Agent3']}."
                    ),
                },
            },
            "required": ["topology_info"],
        }

    def run(
        self,
        agent=None,
        topology_info: Optional[dict] = None,
        **kwargs,
    ) -> dict:
        """执行局部邻居表更新。

        Args:
            agent: 调用该工具的 Agent 实例（由框架自动注入）。
            topology_info: {agent_name: [neighbor_name, ...]} 邻居关系字典。

        Returns:
            {"success": bool, "updated": {...}, "error": str|None}
        """
        if agent is None:
            return {"success": False, "error": "Tool must be invoked with an agent context", "updated": {}}
        if not hasattr(agent, "local_neighbor_table"):
            return {"success": False, "error": "Agent does not support local neighbor table", "updated": {}}
        if topology_info is None:
            return {"success": False, "error": "Missing required argument 'topology_info'", "updated": {}}
        if not isinstance(topology_info, dict):
            return {"success": False, "error": "topology_info must be a dictionary", "updated": {}}

        updated: dict[str, list[str]] = {}
        agent_name = getattr(agent, "name", None)

        for target_name, neighbors in topology_info.items():
            canonical_name = str(target_name).strip()
            # 禁止 Agent 通过工具修改框架维护的自身物理一跳邻居条目
            if agent_name and canonical_name.lower() == agent_name.lower():
                continue
            neighbor_set = set(neighbors) if isinstance(neighbors, (list, set, tuple)) else set()
            agent.local_neighbor_table.setdefault(canonical_name, set()).update(neighbor_set)
            updated[canonical_name] = sorted(agent.local_neighbor_table[canonical_name])

        return {"success": True, "updated": updated, "error": None}
