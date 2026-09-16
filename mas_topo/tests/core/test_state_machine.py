from mas_topo.ledger import IllocutionaryLedger
from mas_topo.state_machine import IllocutionaryStateMachine


def test_directive_becomes_overdue():
    ledger = IllocutionaryLedger()
    ledger.append(1, "A", ["B"], "Directive", "do X", deadline_round=2)
    sm = IllocutionaryStateMachine(directive_timeout=2)
    report = sm.audit(ledger, round_num=4)
    assert len(report.overdue_entries) == 1
    assert report.has_overdue


def test_assertive_closes_without_dispute():
    ledger = IllocutionaryLedger()
    ledger.append(1, "A", ["B"], "Assertive", "x is true")
    sm = IllocutionaryStateMachine()
    report = sm.audit(ledger, round_num=3)
    assert ledger.all_entries()[0].status == "CLOSED"
    assert len(report.closed_entries) == 1


def test_directive_resolved_by_declarative_reply():
    ledger = IllocutionaryLedger()
    ledger.append(1, "A", ["B"], "Directive", "do X")
    ledger.append(2, "B", ["A"], "Declarative", "done X")
    sm = IllocutionaryStateMachine()
    report = sm.audit(ledger, round_num=2)
    assert ledger.get(1).status == "CLOSED"
    assert len(report.closed_entries) == 2


def test_commissive_resolved_by_self_declarative():
    ledger = IllocutionaryLedger()
    ledger.append(1, "A", ["B"], "Commissive", "will do X")
    ledger.append(2, "A", ["B"], "Declarative", "finished X")
    sm = IllocutionaryStateMachine()
    report = sm.audit(ledger, round_num=2)
    assert ledger.get(1).status == "CLOSED"
