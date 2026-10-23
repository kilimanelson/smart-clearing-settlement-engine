"""Shared fixtures for clearing records."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from smart_clearing.schemas import Transaction, TransactionStatus


def make_transaction(**overrides: object) -> Transaction:
    """Build a cleared USD purchase, overriding any field passed by the caller."""
    payload: dict[str, object] = {
        "transaction_id": "T1",
        "cycle_id": "C1",
        "issuer_bank": "A",
        "acquirer_bank": "B",
        "amount_minor": 1000,
        "currency": "USD",
        "status": TransactionStatus.CLEARED,
        "event_timestamp": datetime(2024, 1, 2, 10, 0, tzinfo=UTC),
        "clearing_timestamp": None,
        "reversal_of_transaction_id": None,
    }
    payload.update(overrides)
    return Transaction(**payload)  # type: ignore[arg-type]


@pytest.fixture
def transaction():
    """Factory for validated clearing records."""
    return make_transaction
