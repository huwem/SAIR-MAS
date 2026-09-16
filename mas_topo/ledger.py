"""Illocutionary ledger: shared knowledge base of all pragmatic acts."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LedgerEntry:
    round_num: int
    entry_id: int
    sender: str
    recipients: list[str]
    illocution: str
    content: str
    status: str = "OPEN"  # OPEN | CLOSED | OVERDUE | DISPUTED
    deadline_round: Optional[int] = None
    expected_effect: str = ""
    evidence_entries: list[int] = field(default_factory=list)
    endorsed_by: list[str] = field(default_factory=list)
    disputed_by: list[str] = field(default_factory=list)
    # 语旨行为发生时的全局 answer 快照，用于动态注入当前最佳答案
    answer_snapshot: Optional[str] = None
    # 该 answer_snapshot 对应的验证结果（由框架在 run_python 调用后写入）
    verification_status: Optional[str] = None  # e.g. "PASSED", "FAILED", "ERROR"


class IllocutionaryLedger:
    """In-memory shared ledger with simple query support."""

    def __init__(self):
        self._entries: dict[int, LedgerEntry] = {}
        self._next_id = 1

    def append(
        self,
        round_num: int,
        sender: str,
        recipients: list[str],
        illocution: str,
        content: str,
        status: str = "OPEN",
        deadline_round: Optional[int] = None,
        expected_effect: str = "",
        evidence_entries: Optional[list[int]] = None,
        answer_snapshot: Optional[str] = None,
        verification_status: Optional[str] = None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            round_num=round_num,
            entry_id=self._next_id,
            sender=sender,
            recipients=list(recipients),
            illocution=illocution,
            content=content,
            status=status,
            deadline_round=deadline_round,
            expected_effect=expected_effect,
            evidence_entries=list(evidence_entries) if evidence_entries else [],
            answer_snapshot=answer_snapshot,
            verification_status=verification_status,
        )
        self._entries[entry.entry_id] = entry
        self._next_id += 1
        return entry

    def get(self, entry_id: int) -> Optional[LedgerEntry]:
        return self._entries.get(entry_id)

    def update_status(self, entry_id: int, status: str) -> None:
        entry = self._entries.get(entry_id)
        if entry:
            entry.status = status

    def all_entries(self) -> list[LedgerEntry]:
        return list(self._entries.values())

    def query(
        self,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
        illocution: Optional[str] = None,
        status: Optional[str | list[str]] = None,
        round_min: Optional[int] = None,
    ) -> list[LedgerEntry]:
        results = list(self._entries.values())
        if sender is not None:
            results = [e for e in results if e.sender == sender]
        if recipient is not None:
            results = [e for e in results if recipient in e.recipients]
        if illocution is not None:
            results = [e for e in results if e.illocution == illocution]
        if status is not None:
            if isinstance(status, str):
                status = [status]
            results = [e for e in results if e.status in status]
        if round_min is not None:
            results = [e for e in results if e.round_num >= round_min]
        return results
