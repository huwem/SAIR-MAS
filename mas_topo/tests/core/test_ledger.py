from mas_topo.ledger import IllocutionaryLedger, LedgerEntry


def test_append_and_query():
    ledger = IllocutionaryLedger()
    entry = ledger.append(
        round_num=1,
        sender="Dev",
        recipients=["Tester"],
        illocution="Directive",
        content="test edge cases",
    )
    assert entry.entry_id == 1
    assert entry.status == "OPEN"
    results = ledger.query(recipient="Tester")
    assert len(results) == 1
    assert results[0].sender == "Dev"


def test_status_transition():
    ledger = IllocutionaryLedger()
    e1 = ledger.append(1, "Dev", ["Tester"], "Directive", "test")
    ledger.update_status(e1.entry_id, "CLOSED")
    assert ledger.get(e1.entry_id).status == "CLOSED"


def test_query_by_status_list():
    ledger = IllocutionaryLedger()
    ledger.append(1, "A", ["B"], "Assertive", "x")
    e2 = ledger.append(2, "C", ["D"], "Directive", "y")
    ledger.update_status(e2.entry_id, "OVERDUE")
    results = ledger.query(status=["OPEN", "OVERDUE"])
    assert len(results) == 2


def test_answer_snapshot():
    ledger = IllocutionaryLedger()
    entry = ledger.append(
        1, "Developer", ["Tester"], "Assertive", "Here is the code",
        answer_snapshot="def foo(): return 42",
    )
    assert entry.answer_snapshot == "def foo(): return 42"
    assert ledger.get(entry.entry_id).answer_snapshot == "def foo(): return 42"
