"""AIME 数据集加载器单元测试。"""

import json
from pathlib import Path

import pytest

from mas_topo.datasets.aime_loader import AIMELoader, AIMESample


@pytest.fixture
def aime_jsonl(tmp_path: Path) -> Path:
    """构造一个临时 AIME JSONL 文件，包含 3 道样题。"""
    data = [
        {
            "problem_idx": 1,
            "problem": "What is $2+2$?",
            "answer": 4,
        },
        {
            "problem_idx": 2,
            "problem": "Find the value of $3 \\times 5$.",
            "answer": 15,
        },
        {
            "problem_idx": 3,
            "problem": "Compute $10 - 7$.",
            "answer": 3,
        },
    ]
    file_path = tmp_path / "aime_2024.jsonl"
    with file_path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


@pytest.fixture
def aime_dir(tmp_path: Path) -> Path:
    """构造包含两个 AIME JSONL 文件的临时目录。"""
    for name, data in [
        (
            "aime_2024.jsonl",
            [
                {"problem_idx": 1, "problem": "P1", "answer": 11},
                {"problem_idx": 2, "problem": "P2", "answer": 22},
            ],
        ),
        (
            "aime_2025.jsonl",
            [
                {"problem_idx": 1, "problem": "P3", "answer": 33},
            ],
        ),
    ]:
        file_path = tmp_path / name
        with file_path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    return tmp_path


class TestAIMESample:
    """AIMESample 行为测试。"""

    def test_question_and_ground_truth(self):
        sample = AIMESample(problem="Solve $x+1=2$.", answer=1, index=0)
        assert sample.question == "Solve $x+1=2$."
        assert sample.ground_truth == "1"

    def test_check_answer_correct(self):
        sample = AIMESample(problem="What is $2+2$?", answer=4, index=0)
        assert sample.check_answer("4") is True
        assert sample.check_answer("The answer is 4.") is True
        assert sample.check_answer("Therefore, \\boxed{4}") is True

    def test_check_answer_uses_last_number(self):
        sample = AIMESample(problem="What is $2+2$?", answer=4, index=0)
        # 模型可能在推理过程中提到其他数字，最终答案应为最后一个整数
        assert sample.check_answer("We have 3 apples, then 2+2=4.") is True
        assert sample.check_answer("Maybe 5, no, the answer is 4") is True

    def test_check_answer_wrong(self):
        sample = AIMESample(problem="What is $2+2$?", answer=4, index=0)
        assert sample.check_answer("5") is False
        assert sample.check_answer("The answer is 5.") is False

    def test_check_answer_no_number(self):
        sample = AIMESample(problem="What is $2+2$?", answer=4, index=0)
        assert sample.check_answer("I don't know.") is False
        assert sample.check_answer("") is False

    def test_to_dict(self):
        sample = AIMESample(problem="Long problem " * 100, answer=42, index=7)
        d = sample.to_dict()
        assert d["index"] == 7
        assert d["answer"] == 42
        assert len(d["problem"]) <= 200


class TestAIMELoader:
    """AIMELoader 行为测试。"""

    def test_load_single_file(self, aime_jsonl: Path):
        loader = AIMELoader(str(aime_jsonl))
        samples = loader.load()
        assert len(samples) == 3
        assert samples[0].index == 0
        assert samples[1].index == 1
        assert samples[2].index == 2
        assert samples[0].answer == 4

    def test_load_with_limit(self, aime_jsonl: Path):
        loader = AIMELoader(str(aime_jsonl))
        samples = loader.load(limit=2)
        assert len(samples) == 2
        assert samples[-1].index == 1

    def test_load_via_glob(self, aime_dir: Path):
        loader = AIMELoader(str(aime_dir / "aime_*.jsonl"))
        samples = loader.load()
        assert len(samples) == 3
        assert [s.answer for s in samples] == [11, 22, 33]
        assert list(range(3)) == [s.index for s in samples]

    def test_total_count_single_file(self, aime_jsonl: Path):
        loader = AIMELoader(str(aime_jsonl))
        assert loader.total_count() == 3

    def test_total_count_glob(self, aime_dir: Path):
        loader = AIMELoader(str(aime_dir / "aime_*.jsonl"))
        assert loader.total_count() == 3

    def test_missing_file_raises(self, tmp_path: Path):
        loader = AIMELoader(str(tmp_path / "not_exist.jsonl"))
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_len_after_load(self, aime_jsonl: Path):
        loader = AIMELoader(str(aime_jsonl))
        assert len(loader) == 0
        loader.load()
        assert len(loader) == 3

    def test_get_samples_returns_cached(self, aime_jsonl: Path):
        loader = AIMELoader(str(aime_jsonl))
        samples = loader.load(limit=1)
        assert loader.get_samples() == samples
