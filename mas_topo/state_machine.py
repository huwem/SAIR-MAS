"""Deterministic illocutionary state-transition rules."""

from dataclasses import dataclass, field

from mas_topo.ledger import IllocutionaryLedger, LedgerEntry


@dataclass
class ClosureReport:
    has_overdue: bool = False
    open_entries: list[LedgerEntry] = field(default_factory=list)
    closed_entries: list[LedgerEntry] = field(default_factory=list)
    overdue_entries: list[LedgerEntry] = field(default_factory=list)
    disputed_entries: list[LedgerEntry] = field(default_factory=list)


class IllocutionaryStateMachine:
    """Apply pragmatic state-transition rules to a shared ledger."""

    def __init__(
        self,
        directive_timeout: int = 2,
        commissive_timeout: int = 3,
        assertive_grace: int = 1,
    ):
        self.directive_timeout = directive_timeout
        self.commissive_timeout = commissive_timeout
        self.assertive_grace = assertive_grace

    def audit(self, ledger: IllocutionaryLedger, round_num: int) -> ClosureReport:
        report = ClosureReport()
        entries = ledger.all_entries()

        # Mark overdue first
        for entry in entries:
            if entry.status != "OPEN":
                continue
            deadline = entry.deadline_round
            if entry.illocution == "Directive":
                deadline = deadline or (entry.round_num + self.directive_timeout)
            elif entry.illocution == "Commissive":
                deadline = deadline or (entry.round_num + self.commissive_timeout)

            if deadline and round_num > deadline:
                entry.status = "OVERDUE"

        # Apply per-illocution rules
        for entry in entries:
            if entry.status == "DISPUTED":
                report.disputed_entries.append(entry)
                continue
            if entry.status == "OVERDUE":
                report.overdue_entries.append(entry)
                report.has_overdue = True
                continue
            if entry.status == "CLOSED":
                report.closed_entries.append(entry)
                continue

            # OPEN entry
            if self._is_closed(entry, entries, round_num):
                entry.status = "CLOSED"
                report.closed_entries.append(entry)
            else:
                report.open_entries.append(entry)

        return report

    def _is_closed(self, entry: LedgerEntry, all_entries: list[LedgerEntry], round_num: int) -> bool:
        if entry.illocution == "Expressive":
            return True
        if entry.illocution == "Declarative":
            return True
        if entry.illocution == "Assertive":
            # Close if no dispute within grace period
            return round_num > entry.round_num + self.assertive_grace
        if entry.illocution == "Directive":
            return self._directive_resolved(entry, all_entries)
        if entry.illocution == "Commissive":
            return self._commissive_resolved(entry, all_entries)
        return False

    def _directive_resolved(self, entry: LedgerEntry, all_entries: list[LedgerEntry]) -> bool:
        for later in all_entries:
            if later.entry_id <= entry.entry_id:
                continue
            if later.illocution != "Declarative":
                continue
            # A Declarative reply from any original recipient back to the sender
            # (or broadcast including the sender) resolves the directive.
            if later.sender in entry.recipients and entry.sender in later.recipients:
                return True
        return False

    def _commissive_resolved(self, entry: LedgerEntry, all_entries: list[LedgerEntry]) -> bool:
        for later in all_entries:
            if later.entry_id <= entry.entry_id:
                continue
            if later.illocution != "Declarative":
                continue
            if later.sender == entry.sender:
                return True
        return False
