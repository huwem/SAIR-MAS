"""精确匹配评估器。

适用于 GSM8K/MATH 等需要从预测文本中提取答案并与 ground truth 对比的场景。
支持字符串精确匹配、数字提取匹配、\boxed{} 提取匹配。
自动归一化 LaTeX 与纯文本之间的格式差异。
"""

import re

from mas_topo._registries import evaluator_registry
from mas_topo.datasets.base import DatasetSample
from mas_topo.evaluation.base import Evaluator, EvaluationResult


def _normalize_math_text(text: str) -> str:
    """归一化数学答案文本：消除 LaTeX 与纯文本的格式差异。"""
    if not text:
        return ""
    text = str(text).strip()
    # 去掉 \boxed{...} 包装
    text = re.sub(r"\\boxed\{([^}]*)\}", r"\1", text)
    # \dfrac 同 \frac（先统一，后续 \frac 规则统一处理）
    text = re.sub(r"\\dfrac", r"\\frac", text)
    # LaTeX 分数（带括号）→ 纯文本分数
    text = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", text)
    text = re.sub(r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1/\2", text)
    # LaTeX 分数（不带括号简写）→ 纯文本分数
    text = re.sub(r"\\frac(\d)(\d)", r"\1/\2", text)
    text = re.sub(r"\\frac\s+(\d)(\d)", r"\1/\2", text)
    text = re.sub(r"\\frac(\d)\{([^}]*)\}", r"\1/\2", text)
    text = re.sub(r"\\frac\s+(\d)\{([^}]*)\}", r"\1/\2", text)
    # LaTeX sqrt → Unicode sqrt
    text = re.sub(r"\\sqrt\[(\d+)\]\{([^}]*)\}", r"(\2)^(1/\1)", text)
    text = re.sub(r"\\sqrt\{([^}]*)\}", r"√\1", text)
    # LaTeX text / mbox → plain
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\\mbox\{([^}]*)\}", r"\1", text)
    # LaTeX 希腊字母 → ASCII
    text = re.sub(r"\\pi\b", "pi", text)
    text = re.sub(r"\\theta\b", "theta", text)
    text = re.sub(r"\\alpha\b", "alpha", text)
    text = re.sub(r"\\beta\b", "beta", text)
    # 模型常直接输出 Unicode 希腊字母，统一归一化为 ASCII
    text = text.replace("π", "pi")
    text = text.replace("θ", "theta")
    text = text.replace("α", "alpha")
    text = text.replace("β", "beta")
    # LaTeX 符号 → ASCII
    text = re.sub(r"\\infty", "inf", text)
    text = re.sub(r"\\pm", "+/-", text)
    text = re.sub(r"\\mp", "-/+", text)
    text = re.sub(r"\\cdot", "*", text)
    text = re.sub(r"\\times", "*", text)
    text = re.sub(r"\\cot\b", "cot", text)
    text = re.sub(r"\\in\b", "in", text)
    text = re.sub(r"\\cup\b", "U", text)
    text = re.sub(r"\\cap\b", "∩", text)
    # 去掉 \left, \right
    text = re.sub(r"\\left", "", text)
    text = re.sub(r"\\right", "", text)
    # 去掉 \begin{...} / \end{...} 环境（矩阵/向量）
    text = re.sub(r"\\begin\{[^}]*\}", "", text)
    text = re.sub(r"\\end\{[^}]*\}", "", text)
    # 矩阵内部分隔符归一化：\\ → ,  & → ,
    text = re.sub(r"\\\\", ",", text)
    text = re.sub(r"&", ",", text)
    # 去掉 ^\circ 度数符号
    text = re.sub(r"\^\\circ", "", text)
    text = re.sub(r"\\degree", "", text)
    # 去掉美元符号
    text = re.sub(r"\\?\$", "", text)
    # 去掉 LaTeX 空格命令
    text = re.sub(r"\\[;,]", "", text)
    # 去掉多余空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_boxed(text: str) -> str:
    """从文本中提取 \boxed{...} 中的内容。若找不到则返回原文。"""
    if not text:
        return ""
    # 匹配 \boxed{...}，支持嵌套花括号
    pattern = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return text.strip()


@evaluator_registry.register(
    "exact_match",
    display_name="精确匹配",
    description="字符串或数字精确匹配。支持 extract_number / extract_boxed / normalize_math。",
    config_schema={
        "extract_number": {"type": "bool", "default": False, "description": "从文本中提取数字后再匹配"},
        "extract_boxed": {"type": "bool", "default": False, "description": "从 \boxed{} 中提取答案后再匹配"},
        "normalize_math": {"type": "bool", "default": False, "description": "归一化 LaTeX 与纯文本格式差异"},
    },
)
class ExactMatchEvaluator(Evaluator):
    """精确匹配评估器。

    Args:
        extract_number: 若为 True，则从预测文本和 ground truth 中提取数字后匹配。
        extract_boxed: 若为 True，则从 \boxed{} 中提取答案后匹配。
        normalize_math: 若为 True，则归一化 LaTeX 格式（分数、sqrt、度数等）。
    """

    def __init__(
        self,
        extract_number: bool = False,
        extract_boxed: bool = False,
        normalize_math: bool = False,
        **kwargs,
    ):
        super().__init__(
            extract_number=extract_number,
            extract_boxed=extract_boxed,
            normalize_math=normalize_math,
            **kwargs,
        )
        self.extract_number = extract_number
        self.extract_boxed = extract_boxed
        self.normalize_math = normalize_math

    @staticmethod
    def _extract_last_number(text: str) -> str:
        """从文本中提取最后一个数字。"""
        text = str(text) if text is not None else ""
        numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
        if not numbers:
            return ""
        return numbers[-1].replace(",", "")

    @staticmethod
    def _normalize_orderless_list(text: str) -> str:
        """对裸逗号分隔的无序列表排序后比较。

        若整体被括号/花括号/中括号（含 LaTeX 转义花括号）包裹，则视为有序元组/向量/区间/集合字面量，保持原顺序。
        若所有项都能解析为数字，则按数值排序；否则按字符串排序。
        """
        text = text.strip()
        if len(text) >= 2:
            wrappers = [("(", ")"), ("[", "]"), ("{", "}"), ("\\{", "\\}")]
            for open_ch, close_ch in wrappers:
                if text.startswith(open_ch) and text.endswith(close_ch):
                    return text
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) <= 1:
            return text
        # 尝试按数值排序；若所有项都是数字，则使用数值顺序
        numeric_parts: list[tuple[float, str]] = []
        for p in parts:
            try:
                numeric_parts.append((float(p), p))
            except ValueError:
                break
        if len(numeric_parts) == len(parts):
            return ",".join(p for _, p in sorted(numeric_parts))
        return ",".join(sorted(parts))

    @staticmethod
    def _expand_pm(text: str) -> str:
        r"""将 \\pm 展开为逗号分隔的加/减两个表达式。

        仅处理顶层单一 \\pm，例如 '1 \\pm \\sqrt{19}' → '1+\\sqrt{19}, 1-\\sqrt{19}'。
        """
        text = text.strip()
        # 简单匹配：X \pm Y，其中 X 和 Y 内部不包含 \pm
        if "\\pm" not in text:
            return text
        # 只处理整个答案为一个 \pm 表达式的情况
        match = re.match(r"^(.*?)\\pm(.*)$", text)
        if not match:
            return text
        left = match.group(1).strip()
        right = match.group(2).strip()
        if not left or not right:
            return text
        return f"{left}+{right}, {left}-{right}"

    def _clean(self, text: str) -> str:
        """对答案文本做清洗流水线。"""
        t = str(text) if text is not None else ""
        if self.extract_boxed:
            t = _extract_boxed(t)
        # ── LaTeX 格式归一化（不影响数学含义，始终执行）──
        # 格式化包装：\left, \right, \dfrac
        t = re.sub(r"\\left", "", t)
        t = re.sub(r"\\right", "", t)
        t = re.sub(r"\\dfrac", r"\\frac", t)
        # 度数符号（支持 ^\circ 与 ^{\circ} 两种写法）
        t = re.sub(r"\^(?:\\circ|\{\\circ\})", "", t)
        t = re.sub(r"\\degree", "", t)
        # 文本包装 → 提取内部文本
        t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
        t = re.sub(r"\\mbox\{([^}]*)\}", r"\1", t)
        # 矩阵/向量环境包装 → 去除，分隔符归一化
        t = re.sub(r"\\begin\{[^}]*\}", "", t)
        t = re.sub(r"\\end\{[^}]*\}", "", t)
        t = re.sub(r"\\\\", ",", t)
        t = re.sub(r"&", ",", t)
        # 数学模式标记与间距命令
        t = re.sub(r"\\?\$", "", t)
        t = re.sub(r"\\[;,]", "", t)
        t = re.sub(r"\\qquad|\\quad", "", t)
        # 百分比符号去除（模型常输出 10\%，ground truth 为 10）
        t = re.sub(r"(\d)\\%", r"\1", t)
        t = re.sub(r"(\d)%", r"\1", t)
        # 单字符可选花括号归一化（上标/根号等）：\sqrt{2} ↔ \sqrt2，\sqrt{x} ↔ \sqrt x
        # 排除 \frac，避免 \frac{1}{3} 被错误地归一化为 \frac1{3}
        t = re.sub(r"\\(?!frac\b)([a-zA-Z]+)\{([a-zA-Z0-9])\}", r"\\\1\2", t)
        # 单字符下标花括号归一化：2516_{8} ↔ 2516_8
        t = re.sub(r"_\{([a-zA-Z0-9])\}", r"_\1", t)
        # 简写分数归一化：\frac43 → \frac{4}{3}
        t = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", t)
        # 行内分数 LaTeX 化：-1/3 → -\frac{1}{3}
        t = re.sub(r"(-?\d+)\s*/\s*(\d+)", r"\\frac{\1}{\2}", t)
        # 将分子中的负号提到分数前面：\frac{-1}{3} → -\frac{1}{3}
        t = re.sub(r"\\frac\{(-)([^}]+)\}\{([^}]+)\}", r"-\\frac{\2}{\3}", t)
        # 简单变量赋值前缀去除：x=5, $x = 5$, a=83, k=2 等
        t = re.sub(r"^[a-zA-Z]\s*=\s*", "", t)
        # \pm 展开：1 \pm \sqrt{19} → 1+\sqrt{19}, 1-\sqrt{19}
        t = self._expand_pm(t)
        if self.normalize_math:
            t = _normalize_math_text(t)
        if self.extract_number:
            # 仅对纯文本（无 LaTeX 命令）提取数字；含 LaTeX 结构的表达式
            # 提取"最后一个数字"会丢失数学含义
            # （如 \frac{23}{3} → 3、(5, \frac{\pi}{2}) → 2 的假阳性）
            if "\\" not in t:
                t = self._extract_last_number(t)
        # 无序列表排序（仅对裸逗号分隔值）
        t = self._normalize_orderless_list(t)
        # 最终归一化：去空格、统一小写
        t = re.sub(r"\s+", "", t).lower()
        return t

    def evaluate(self, sample: DatasetSample, predicted: str) -> EvaluationResult:
        pred = str(predicted) if predicted is not None else ""
        gt = str(sample.ground_truth)

        pred_clean = self._clean(pred)
        gt_clean = self._clean(gt)

        is_correct = pred_clean == gt_clean and pred_clean != ""

        return EvaluationResult(
            is_correct=is_correct,
            predicted_answer=pred,
            metric_value=1.0 if is_correct else 0.0,
            metric_name="exact_match",
            details={
                "extract_number": self.extract_number,
                "extract_boxed": self.extract_boxed,
                "normalize_math": self.normalize_math,
                "predicted_clean": pred_clean,
                "ground_truth_clean": gt_clean,
            },
        )
