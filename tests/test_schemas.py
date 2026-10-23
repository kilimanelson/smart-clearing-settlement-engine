"""Contracts reject ambiguous money and format minor units for display only."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from smart_clearing.schemas import TransactionCreate, TransactionStatus, format_minor_units
from tests.conftest import make_transaction


def test_amount_must_be_an_integer_minor_unit() -> None:
    with pytest.raises(ValidationError):
        make_transaction(amount_minor=10.5)
    with pytest.raises(ValidationError):
        make_transaction(amount_minor=True)


def test_timestamps_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        make_transaction(event_timestamp=datetime(2024, 1, 2, 10, 0))


def test_ingestion_rejects_same_bank_and_non_positive_amounts() -> None:
    with pytest.raises(ValidationError):
        TransactionCreate(
            transaction_id="T1",
            cycle_id="C1",
            issuer_bank="A",
            acquirer_bank="A",
            amount_minor=100,
            currency="USD",
            status=TransactionStatus.CLEARED,
            event_timestamp=datetime(2024, 1, 2, 10, tzinfo=UTC),
        )
    with pytest.raises(ValidationError):
        TransactionCreate(
            transaction_id="T1",
            cycle_id="C1",
            issuer_bank="A",
            acquirer_bank="B",
            amount_minor=0,
            currency="USD",
            status=TransactionStatus.CLEARED,
            event_timestamp=datetime(2024, 1, 2, 10, tzinfo=UTC),
        )


def test_reversed_ingestion_requires_a_parent_id() -> None:
    with pytest.raises(ValidationError):
        TransactionCreate(
            transaction_id="T2",
            cycle_id="C1",
            issuer_bank="A",
            acquirer_bank="B",
            amount_minor=100,
            currency="USD",
            status=TransactionStatus.REVERSED,
            event_timestamp=datetime(2024, 1, 2, 11, tzinfo=UTC),
        )


def test_format_minor_units_is_display_only() -> None:
    assert format_minor_units(0, "USD") == "0.00 USD"
    assert format_minor_units(100, "USD") == "1.00 USD"
    assert format_minor_units(1099, "USD") == "10.99 USD"
    assert format_minor_units(-900, "USD") == "-9.00 USD"
