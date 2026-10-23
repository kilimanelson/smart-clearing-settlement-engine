"""Deterministic deferred net settlement for one currency and one cycle.

Sign convention
---------------
A negative net position means the participant owes money. A positive net
position means the participant is due to receive money. For every report this
module returns, the participant net positions sum to zero.

Inclusion rules, in precedence order
------------------------------------
1. Duplicate ``transaction_id`` values reject the whole submission.
2. ``amount_minor <= 0`` is quarantined.
3. An issuer equal to the acquirer is quarantined.
4. A different ``cycle_id`` is quarantined.
5. A different currency is quarantined.
6. ``pending`` records are quarantined.
7. ``cleared`` records that point at another transaction are quarantined.
8. ``reversed`` records without ``reversal_of_transaction_id`` are quarantined.
9. A remaining ``cleared`` record creates an obligation from the issuer to the
   acquirer. A remaining ``reversed`` record is not a new purchase: it creates
   the opposite obligation for the same banks and amount. The referenced
   original does not have to be in the same submission; a reversal can settle
   in a later cycle.

Fees and interchange are out of scope. All sums use Python integers.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from typing import Never

from smart_clearing.schemas import (
    BilateralObligation,
    ClearingReport,
    NetPosition,
    QuarantinedRecord,
    QuarantineReason,
    Transaction,
    TransactionStatus,
)

logger = logging.getLogger(__name__)


class ClearingError(Exception):
    """Base class for clearing failures that a caller can handle."""


class DuplicateTransactionError(ClearingError):
    """The submission contains the same transaction id more than once."""

    def __init__(self, transaction_ids: list[str]) -> None:
        self.transaction_ids = transaction_ids
        super().__init__(f"Duplicate transaction IDs: {', '.join(transaction_ids)}")


class ReconciliationError(ClearingError):
    """A position set breaks conservation of obligations."""

    def __init__(self, cycle_id: str, detail: str) -> None:
        self.cycle_id = cycle_id
        super().__init__(f"Cycle {cycle_id} failed reconciliation: {detail}")


def _assert_never(value: Never) -> None:
    raise ClearingError(f"Unhandled transaction status: {value}")


def _quarantine(record: Transaction, reason: QuarantineReason, detail: str) -> QuarantinedRecord:
    return QuarantinedRecord(
        transaction_id=record.transaction_id, reason_code=reason, detail=detail
    )


def classify_transaction(
    record: Transaction,
    *,
    cycle_id: str,
    currency: str,
) -> QuarantinedRecord | TransactionStatus:
    """Return ``CLEARED`` or ``REVERSED`` when the record enters the cycle.

    Any other return value is a quarantine decision. The first matching rule
    wins, so a negative amount in the wrong cycle is reported as non-positive.
    """
    if record.amount_minor <= 0:
        return _quarantine(
            record,
            QuarantineReason.NON_POSITIVE_AMOUNT,
            "Amount must be a positive number of minor units.",
        )
    if record.issuer_bank == record.acquirer_bank:
        return _quarantine(
            record,
            QuarantineReason.SAME_BANK,
            "Issuer and acquirer must be different participants.",
        )
    if record.cycle_id != cycle_id:
        return _quarantine(
            record,
            QuarantineReason.WRONG_CYCLE,
            f"Record belongs to {record.cycle_id}, not {cycle_id}.",
        )
    if record.currency != currency:
        return _quarantine(
            record,
            QuarantineReason.WRONG_CURRENCY,
            f"Record currency {record.currency} is outside this {currency} cycle.",
        )
    status = record.status
    if status == TransactionStatus.PENDING:
        return _quarantine(
            record,
            QuarantineReason.NOT_ELIGIBLE_STATUS,
            "Pending records are not eligible for settlement.",
        )
    if status == TransactionStatus.CLEARED:
        if record.reversal_of_transaction_id is not None:
            return _quarantine(
                record,
                QuarantineReason.CLEARED_HAS_REVERSAL_REFERENCE,
                "A cleared purchase cannot reference a reversal parent.",
            )
        return TransactionStatus.CLEARED
    if status == TransactionStatus.REVERSED:
        if not record.reversal_of_transaction_id:
            return _quarantine(
                record,
                QuarantineReason.REVERSAL_MISSING_REFERENCE,
                "A reversal must identify the transaction it reverses.",
            )
        return TransactionStatus.REVERSED
    _assert_never(status)


def enforce_zero_sum(
    cycle_id: str,
    positions: Sequence[NetPosition],
    obligations: Sequence[BilateralObligation],
) -> None:
    """Raise ``ReconciliationError`` unless obligations are conserved.

    The three checks are equivalent in a correct bilateral netting, and each
    one is explicit so a future edit cannot drop the cycle-level identity.
    """
    net_sum = sum((position.net_position_minor for position in positions), 0)
    if net_sum != 0:
        raise ReconciliationError(cycle_id, f"net positions sum to {net_sum}, expected 0")
    gross_debit = sum((position.gross_debit_minor for position in positions), 0)
    gross_credit = sum((position.gross_credit_minor for position in positions), 0)
    if gross_debit != gross_credit:
        raise ReconciliationError(
            cycle_id,
            f"gross debit {gross_debit} does not equal gross credit {gross_credit}",
        )
    bilateral = sum((item.amount_minor for item in obligations), 0)
    if bilateral != gross_debit:
        raise ReconciliationError(
            cycle_id,
            f"bilateral obligations {bilateral} do not equal gross debit {gross_debit}",
        )
    for position in positions:
        expected = position.gross_credit_minor - position.gross_debit_minor
        if position.net_position_minor != expected:
            raise ReconciliationError(
                cycle_id,
                f"{position.bank_id} net position does not equal credit minus debit",
            )


def clear_cycle(
    records: Sequence[Transaction],
    *,
    cycle_id: str,
    currency: str,
    participants: Sequence[str] | None = None,
) -> ClearingReport:
    """Clear one cycle and return a reconciled report.

    ``participants`` forces a zero position for banks with no included
    activity. The same records in any order produce the same report.
    """
    if len(currency) != 3 or not currency.isalpha() or currency != currency.upper():
        raise ClearingError("currency must be three uppercase letters")
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[record.transaction_id] += 1
    duplicates = sorted(transaction_id for transaction_id, count in counts.items() if count > 1)
    if duplicates:
        logger.warning("Duplicate transaction ids in cycle %s: %s", cycle_id, duplicates)
        raise DuplicateTransactionError(duplicates)

    participant_ids: set[str] = set()
    if participants is not None:
        for bank_id in participants:
            if not isinstance(bank_id, str) or not bank_id:
                raise ClearingError("Participant ids must be non-empty strings")
            participant_ids.add(bank_id)

    quarantined: list[QuarantinedRecord] = []
    flows: dict[tuple[str, str], int] = defaultdict(int)
    purchase_count = 0
    reversal_count = 0
    for record in records:
        decision = classify_transaction(record, cycle_id=cycle_id, currency=currency)
        if isinstance(decision, QuarantinedRecord):
            quarantined.append(decision)
            continue
        if decision == TransactionStatus.CLEARED:
            flows[(record.issuer_bank, record.acquirer_bank)] += record.amount_minor
            purchase_count += 1
            continue
        if decision == TransactionStatus.REVERSED:
            flows[(record.acquirer_bank, record.issuer_bank)] += record.amount_minor
            reversal_count += 1
            continue
        _assert_never(decision)

    pair_keys = {tuple(sorted((debtor, creditor))) for debtor, creditor in flows}
    obligations: list[BilateralObligation] = []
    for bank_a, bank_b in sorted(pair_keys):
        forward = flows.get((bank_a, bank_b), 0)
        backward = flows.get((bank_b, bank_a), 0)
        net_amount = forward - backward
        if net_amount > 0:
            obligations.append(
                BilateralObligation(
                    debtor_bank=bank_a,
                    creditor_bank=bank_b,
                    amount_minor=net_amount,
                    currency=currency,
                )
            )
        elif net_amount < 0:
            obligations.append(
                BilateralObligation(
                    debtor_bank=bank_b,
                    creditor_bank=bank_a,
                    amount_minor=-net_amount,
                    currency=currency,
                )
            )

    gross_debit: dict[str, int] = defaultdict(int)
    gross_credit: dict[str, int] = defaultdict(int)
    for obligation in obligations:
        gross_debit[obligation.debtor_bank] += obligation.amount_minor
        gross_credit[obligation.creditor_bank] += obligation.amount_minor

    bank_ids = set(gross_debit) | set(gross_credit) | participant_ids
    positions = [
        NetPosition(
            bank_id=bank_id,
            net_position_minor=gross_credit[bank_id] - gross_debit[bank_id],
            gross_debit_minor=gross_debit[bank_id],
            gross_credit_minor=gross_credit[bank_id],
            currency=currency,
        )
        for bank_id in sorted(bank_ids)
    ]
    quarantined.sort(key=lambda item: (item.transaction_id, item.reason_code.value))
    enforce_zero_sum(cycle_id, positions, obligations)
    net_sum = sum((position.net_position_minor for position in positions), 0)
    logger.info(
        "Cleared cycle %s currency %s purchases=%d reversals=%d quarantined=%d net_sum=%d",
        cycle_id,
        currency,
        purchase_count,
        reversal_count,
        len(quarantined),
        net_sum,
    )
    return ClearingReport(
        cycle_id=cycle_id,
        currency=currency,
        purchase_count=purchase_count,
        reversal_count=reversal_count,
        quarantined_count=len(quarantined),
        bilateral_obligations=obligations,
        net_positions=positions,
        net_position_sum_minor=net_sum,
        reconciled=True,
        quarantined=quarantined,
    )
