"""Deterministic DNS clearing, including invalid and adversarial batches."""

from __future__ import annotations

import random

import pytest

from smart_clearing.clearing import (
    ClearingError,
    DuplicateTransactionError,
    ReconciliationError,
    clear_cycle,
    enforce_zero_sum,
)
from smart_clearing.schemas import NetPosition, QuarantineReason, TransactionStatus
from tests.conftest import make_transaction


def _cleared(records, **kwargs):
    report = clear_cycle(records, cycle_id="C1", currency="USD", **kwargs)
    return {
        position.bank_id: position.net_position_minor for position in report.net_positions
    }, report


def test_three_bank_cycle_has_known_positions() -> None:
    records = [
        make_transaction(
            transaction_id="T1", issuer_bank="A", acquirer_bank="B", amount_minor=1000
        ),
        make_transaction(transaction_id="T2", issuer_bank="B", acquirer_bank="C", amount_minor=400),
        make_transaction(transaction_id="T3", issuer_bank="C", acquirer_bank="A", amount_minor=100),
    ]
    nets, report = _cleared(records)
    # C receives 400 from B and owes 100 to A, so C is a net creditor.
    assert nets == {"A": -900, "B": 600, "C": 300}
    assert report.net_position_sum_minor == 0
    assert report.reconciled is True
    assert report.purchase_count == 3
    obligations = {
        (item.debtor_bank, item.creditor_bank): item.amount_minor
        for item in report.bilateral_obligations
    }
    assert obligations == {("A", "B"): 1000, ("B", "C"): 400, ("C", "A"): 100}


def test_multiple_transactions_between_the_same_banks_are_summed() -> None:
    records = [
        make_transaction(transaction_id="T1", amount_minor=1000),
        make_transaction(transaction_id="T2", amount_minor=500),
    ]
    nets, _report = _cleared(records)
    assert nets == {"A": -1500, "B": 1500}


def test_other_cycles_and_currencies_are_quarantined() -> None:
    records = [
        make_transaction(transaction_id="T1", amount_minor=1000),
        make_transaction(transaction_id="T2", cycle_id="C2", amount_minor=9999),
        make_transaction(transaction_id="T3", currency="EUR", amount_minor=8888),
    ]
    nets, report = _cleared(records)
    assert nets == {"A": -1000, "B": 1000}
    reasons = {item.reason_code for item in report.quarantined}
    assert reasons == {QuarantineReason.WRONG_CYCLE, QuarantineReason.WRONG_CURRENCY}


def test_pending_records_are_excluded_and_reversals_offset_the_original() -> None:
    records = [
        make_transaction(transaction_id="T1", amount_minor=1000),
        make_transaction(transaction_id="T2", amount_minor=5000, status=TransactionStatus.PENDING),
        make_transaction(
            transaction_id="T3",
            amount_minor=400,
            status=TransactionStatus.REVERSED,
            reversal_of_transaction_id="T1",
        ),
    ]
    nets, report = _cleared(records)
    assert nets == {"A": -600, "B": 600}
    assert report.purchase_count == 1
    assert report.reversal_count == 1
    assert report.quarantined[0].reason_code == QuarantineReason.NOT_ELIGIBLE_STATUS


def test_duplicate_transaction_ids_reject_the_batch() -> None:
    records = [
        make_transaction(transaction_id="T1", amount_minor=1000),
        make_transaction(transaction_id="T1", amount_minor=1000),
    ]
    with pytest.raises(DuplicateTransactionError) as caught:
        clear_cycle(records, cycle_id="C1", currency="USD")
    assert caught.value.transaction_ids == ["T1"]


def test_zero_and_negative_amounts_are_quarantined() -> None:
    records = [
        make_transaction(transaction_id="T1", amount_minor=1000),
        make_transaction(transaction_id="T2", amount_minor=0),
        make_transaction(transaction_id="T3", amount_minor=-50),
        make_transaction(transaction_id="T4", issuer_bank="A", acquirer_bank="A", amount_minor=20),
    ]
    nets, report = _cleared(records)
    assert nets == {"A": -1000, "B": 1000}
    assert {item.reason_code for item in report.quarantined} == {
        QuarantineReason.NON_POSITIVE_AMOUNT,
        QuarantineReason.SAME_BANK,
    }


def test_bank_with_no_activity_is_reported_as_flat() -> None:
    nets, report = _cleared([make_transaction()], participants=["A", "B", "D"])
    assert nets["D"] == 0
    assert report.net_position_sum_minor == 0


def test_empty_cycle_reconciles() -> None:
    report = clear_cycle([], cycle_id="C1", currency="USD")
    assert report.reconciled is True
    assert report.net_positions == []
    assert report.net_position_sum_minor == 0
    assert report.purchase_count == 0


def test_failed_reconciliation_is_a_controlled_error() -> None:
    positions = [
        NetPosition(
            bank_id="A",
            net_position_minor=10,
            gross_debit_minor=0,
            gross_credit_minor=10,
            currency="USD",
        ),
        NetPosition(
            bank_id="B",
            net_position_minor=-5,
            gross_debit_minor=5,
            gross_credit_minor=0,
            currency="USD",
        ),
    ]
    with pytest.raises(ReconciliationError, match="sum to 5"):
        enforce_zero_sum("C1", positions, [])


def test_clear_cycle_always_checks_reconciliation(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"count": 0}
    original = enforce_zero_sum

    def _wrapped(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr("smart_clearing.clearing.enforce_zero_sum", _wrapped)
    clear_cycle([make_transaction()], cycle_id="C1", currency="USD")
    assert called["count"] == 1


def test_same_records_in_any_order_produce_the_same_report() -> None:
    records = [
        make_transaction(
            transaction_id="T1", issuer_bank="A", acquirer_bank="B", amount_minor=1000
        ),
        make_transaction(transaction_id="T2", issuer_bank="B", acquirer_bank="C", amount_minor=400),
        make_transaction(transaction_id="T3", issuer_bank="C", acquirer_bank="A", amount_minor=250),
    ]
    forward = clear_cycle(records, cycle_id="C1", currency="USD")
    backward = clear_cycle(list(reversed(records)), cycle_id="C1", currency="USD")
    assert forward.model_dump(mode="json") == backward.model_dump(mode="json")


def test_random_valid_batches_always_reconcile() -> None:
    generator = random.Random(0)
    banks = ["A", "B", "C", "D"]
    for batch in range(25):
        records = []
        for index in range(generator.randint(0, 30)):
            issuer, acquirer = generator.sample(banks, 2)
            status_roll = generator.random()
            if status_roll < 0.15:
                status = TransactionStatus.PENDING
                parent = None
            elif status_roll < 0.25:
                status = TransactionStatus.REVERSED
                parent = f"ORIG-{batch}-{index}"
            else:
                status = TransactionStatus.CLEARED
                parent = None
            amount = generator.choice([0, -5, generator.randint(1, 5000)])
            records.append(
                make_transaction(
                    transaction_id=f"{batch}-{index}",
                    issuer_bank=issuer,
                    acquirer_bank=acquirer,
                    amount_minor=amount,
                    status=status,
                    reversal_of_transaction_id=parent,
                )
            )
        report = clear_cycle(records, cycle_id="C1", currency="USD", participants=banks)
        assert report.reconciled is True
        assert report.net_position_sum_minor == 0
        assert sum(position.gross_debit_minor for position in report.net_positions) == sum(
            item.amount_minor for item in report.bilateral_obligations
        )


def test_reversal_without_a_reference_is_quarantined() -> None:
    _positions, report = _cleared(
        [
            make_transaction(
                transaction_id="T9",
                status=TransactionStatus.REVERSED,
                reversal_of_transaction_id=None,
            )
        ]
    )
    assert report.reversal_count == 0
    assert report.quarantined[0].reason_code == QuarantineReason.REVERSAL_MISSING_REFERENCE


def test_invalid_currency_code_is_rejected() -> None:
    with pytest.raises(ClearingError):
        clear_cycle([], cycle_id="C1", currency="usd")
