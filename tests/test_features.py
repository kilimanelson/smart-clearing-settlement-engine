"""Feature columns, leakage checks, and chronological splits."""

from datetime import UTC, datetime

import pandas as pd
import pytest

from smart_clearing.features import (
    MODEL_FEATURES,
    LeakageError,
    assert_no_leakage,
    chronological_cycle_split,
    compute_observed_features,
    cumulative_intraday_fraction,
    feature_matrix,
    forecast_net_debit_minor,
)
from smart_clearing.schemas import TransactionStatus
from tests.conftest import make_transaction


def test_intraday_prior_is_monotonic_and_does_not_read_transactions() -> None:
    start = datetime(2024, 1, 2, 8, 0, tzinfo=UTC)
    middle = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
    end = datetime(2024, 1, 2, 17, 0, tzinfo=UTC)
    assert cumulative_intraday_fraction(start) == pytest.approx(0.05)
    assert cumulative_intraday_fraction(end) == pytest.approx(1.0)
    assert (
        cumulative_intraday_fraction(start)
        < cumulative_intraday_fraction(middle)
        < cumulative_intraday_fraction(end)
    )


def test_forecast_uses_only_the_supplied_volumes() -> None:
    moment = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
    forecast = forecast_net_debit_minor(10_000, 0, moment)
    assert forecast > 10_000
    assert forecast_net_debit_minor(0, 10_000, moment) == 0


def test_leakage_columns_are_rejected() -> None:
    with pytest.raises(LeakageError):
        assert_no_leakage(["actual_shortfall_at_cutoff", "outgoing_volume_so_far"])


def test_feature_matrix_keeps_only_pre_cutoff_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                **{name: 1 for name in MODEL_FEATURES},
                "actual_shortfall_at_cutoff": 1,
                "actual_final_net_debit": 50,
                "available_liquidity_at_cutoff": 10,
            }
        ]
    )
    matrix = feature_matrix(frame)
    assert list(matrix.columns) == list(MODEL_FEATURES)


def test_future_record_cannot_enter_the_feature_function() -> None:
    prediction = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
    records = [
        make_transaction(event_timestamp=datetime(2024, 1, 2, 16, 0, tzinfo=UTC)),
    ]
    with pytest.raises(LeakageError):
        compute_observed_features(
            records,
            cycle_id="C1",
            currency="USD",
            participants=["A", "B"],
            prediction_timestamp=prediction,
            historical_rates={"A": 0.0, "B": 0.0},
            liquidity_at_prediction={"A": 100, "B": 100},
        )


def test_observed_features_ignore_pending_amounts_in_the_obligation() -> None:
    prediction = datetime(2024, 1, 2, 14, 0, tzinfo=UTC)
    records = [
        make_transaction(transaction_id="T1", amount_minor=1_000),
        make_transaction(transaction_id="T2", amount_minor=9_000, status=TransactionStatus.PENDING),
    ]
    rows = compute_observed_features(
        records,
        cycle_id="C1",
        currency="USD",
        participants=["A", "B"],
        prediction_timestamp=prediction,
        historical_rates={"A": 0.25, "B": 0.0},
        liquidity_at_prediction={"A": 500, "B": 500},
    )
    by_bank = {row["bank_id"]: row for row in rows}
    assert by_bank["A"]["outgoing_volume_so_far"] == 1_000
    assert by_bank["A"]["transaction_count_so_far"] == 2
    assert by_bank["A"]["historical_shortfall_rate"] == 0.25
    assert by_bank["A"]["predicted_or_forecast_net_debit"] > 1_000


def test_cycles_stay_together_and_in_time_order() -> None:
    rows = []
    for day in range(1, 7):
        for bank in ("A", "B"):
            rows.append(
                {
                    "cycle_id": f"CY-2024010{day}",
                    "bank_id": bank,
                    "actual_shortfall_at_cutoff": day % 2,
                    **{name: 0 for name in MODEL_FEATURES},
                }
            )
    train, validation, test = chronological_cycle_split(pd.DataFrame(rows))
    assert max(train["cycle_id"]) < min(validation["cycle_id"])
    assert max(validation["cycle_id"]) < min(test["cycle_id"])
    assert set(train["cycle_id"]).isdisjoint(test["cycle_id"])
    labeled = pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ]
    )
    for _cycle_id, group in labeled.groupby("cycle_id"):
        assert group["split"].nunique() == 1
