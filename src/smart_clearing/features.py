"""Pre-cutoff features and chronological dataset splits.

Model features are the columns in ``MODEL_FEATURES`` and nothing else. Labels,
cutoff liquidity, scenario tags, and identifiers are retained on the snapshot
table for evaluation, then removed before a model is fit.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

import pandas as pd

from smart_clearing.clearing import clear_cycle
from smart_clearing.schemas import Transaction

logger = logging.getLogger(__name__)

MODEL_FEATURES: tuple[str, ...] = (
    "incoming_volume_so_far",
    "outgoing_volume_so_far",
    "transaction_count_so_far",
    "average_transaction_amount_so_far",
    "recent_volume_change",
    "historical_shortfall_rate",
    "available_liquidity_at_prediction",
    "predicted_or_forecast_net_debit",
)

LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "actual_shortfall_at_cutoff",
        "actual_final_net_debit",
        "available_liquidity_at_cutoff",
        "funding_shock_after_prediction",
        "scenario",
        "liquidity_at_cutoff",
    }
)


class FeatureError(Exception):
    """The snapshot table cannot be turned into a valid feature matrix."""


class LeakageError(FeatureError):
    """A feature matrix contains information that was not known at prediction time."""


def cumulative_intraday_fraction(
    moment: datetime,
    *,
    start_hour: int = 8,
    end_hour: int = 17,
) -> float:
    """Return a fixed prior for how much of the day's value has usually arrived.

    This curve is a documented simulation assumption. It is not estimated from
    the cycle being forecast, so it does not read transactions after ``moment``.
    """
    if start_hour >= end_hour:
        raise ValueError("intraday start_hour must be earlier than end_hour")
    minutes = moment.hour * 60 + moment.minute
    start = start_hour * 60
    end = end_hour * 60
    if minutes <= start:
        return 0.05
    if minutes >= end:
        return 1.0
    progress = (minutes - start) / (end - start)
    return 0.05 + 0.95 * (1.0 - (1.0 - progress) ** 1.4)


def scale_observed_to_day(observed_minor: int, fraction: float) -> int:
    """Scale an observed minor-unit volume by the intraday prior.

    The fraction is converted to basis points once. The returned volume is an
    integer. It is a forecast input, not a settled obligation.
    """
    if isinstance(observed_minor, bool) or not isinstance(observed_minor, int):
        raise TypeError("observed volume must be an int")
    if observed_minor < 0:
        raise ValueError("observed volume cannot be negative")
    if observed_minor == 0:
        return 0
    basis_points = max(1, int(round(fraction * 10_000)))
    return observed_minor * 10_000 // basis_points


def forecast_net_debit_minor(
    outgoing_minor: int,
    incoming_minor: int,
    moment: datetime,
) -> int:
    """Forecast the positive amount a bank will owe at the end of the day.

    Incoming and outgoing volumes must already be limited to records observed
    at ``moment``. Creditor forecasts return zero because they are not a debit.
    """
    fraction = cumulative_intraday_fraction(moment)
    incoming = scale_observed_to_day(incoming_minor, fraction)
    outgoing = scale_observed_to_day(outgoing_minor, fraction)
    net_position = incoming - outgoing
    if net_position < 0:
        return -net_position
    return 0


def assert_no_leakage(columns: Sequence[str]) -> None:
    """Raise when a model matrix includes a forbidden column."""
    overlap = sorted(set(columns).intersection(LEAKAGE_COLUMNS))
    if overlap:
        raise LeakageError(f"Leakage columns present: {overlap}")
    for column in columns:
        lowered = column.lower()
        if lowered.startswith("actual_") or "cutoff" in lowered or lowered.startswith("label"):
            raise LeakageError(f"Column {column} is not available at prediction time")


def compute_observed_features(
    records: Sequence[Transaction],
    *,
    cycle_id: str,
    currency: str,
    participants: Sequence[str],
    prediction_timestamp: datetime,
    historical_rates: Mapping[str, float],
    liquidity_at_prediction: Mapping[str, int],
) -> list[dict[str, object]]:
    """Build one feature row per participant from records already observed.

    The function rejects any record timestamped after ``prediction_timestamp``.
    Callers that want a later label must pass a separate record set to the
    clearing engine. This function does not calculate that label.
    """
    future = [
        record.transaction_id for record in records if record.event_timestamp > prediction_timestamp
    ]
    if future:
        raise LeakageError(
            "Feature records include events after the prediction timestamp: "
            + ", ".join(sorted(future))
        )
    report = clear_cycle(
        records,
        cycle_id=cycle_id,
        currency=currency,
        participants=participants,
    )
    positions = {position.bank_id: position for position in report.net_positions}
    recent_start = prediction_timestamp - timedelta(hours=1)
    prior_start = prediction_timestamp - timedelta(hours=2)
    rows: list[dict[str, object]] = []
    for bank_id in sorted(participants):
        position = positions[bank_id]
        party_amounts = [
            (record.event_timestamp, record.amount_minor)
            for record in records
            if record.amount_minor > 0
            and (record.issuer_bank == bank_id or record.acquirer_bank == bank_id)
        ]
        count = len(party_amounts)
        total_amount = sum(amount for _, amount in party_amounts)
        average = float(total_amount) / count if count else 0.0
        recent = sum(
            amount
            for stamp, amount in party_amounts
            if recent_start < stamp <= prediction_timestamp
        )
        prior = sum(
            amount for stamp, amount in party_amounts if prior_start < stamp <= recent_start
        )
        if prior == 0:
            recent_change = 0.0 if recent == 0 else 1.0
        else:
            recent_change = (recent - prior) / prior
        forecast_debit = forecast_net_debit_minor(
            position.gross_debit_minor,
            position.gross_credit_minor,
            prediction_timestamp,
        )
        rows.append(
            {
                "bank_id": bank_id,
                "incoming_volume_so_far": position.gross_credit_minor,
                "outgoing_volume_so_far": position.gross_debit_minor,
                "transaction_count_so_far": count,
                "average_transaction_amount_so_far": average,
                "recent_volume_change": recent_change,
                "historical_shortfall_rate": float(historical_rates.get(bank_id, 0.0)),
                "available_liquidity_at_prediction": int(liquidity_at_prediction[bank_id]),
                "predicted_or_forecast_net_debit": forecast_debit,
            }
        )
    assert_no_leakage(MODEL_FEATURES)
    logger.debug("Computed pre-cutoff features for cycle %s banks=%d", cycle_id, len(rows))
    return rows


def feature_matrix(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Return model columns only, in a stable order."""
    missing = [column for column in MODEL_FEATURES if column not in snapshots.columns]
    if missing:
        raise FeatureError(f"Snapshot is missing feature columns: {missing}")
    frame = snapshots.loc[:, list(MODEL_FEATURES)].copy()
    assert_no_leakage(frame.columns)
    if frame.isna().any().any():
        raise FeatureError("Feature matrix contains missing values")
    return frame


def chronological_cycle_split(
    snapshots: pd.DataFrame,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by whole cycles, keeping earlier cycles in earlier splits.

    Random row splits are intentionally unavailable. Every bank in a cycle
    stays in the same split because the cut is on ``cycle_id``.
    """
    if "cycle_id" not in snapshots.columns:
        raise FeatureError("snapshots must include cycle_id")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a non-empty test share")
    ordered = snapshots.sort_values(["cycle_id", "bank_id"]).reset_index(drop=True)
    cycles = list(dict.fromkeys(ordered["cycle_id"].tolist()))
    if len(cycles) < 3:
        raise FeatureError("At least three cycles are required for train, validation, and test")
    train_count = max(1, int(len(cycles) * train_fraction))
    validation_count = max(1, int(len(cycles) * validation_fraction))
    while train_count + validation_count >= len(cycles):
        if validation_count > 1:
            validation_count -= 1
        elif train_count > 1:
            train_count -= 1
        else:
            raise FeatureError("Unable to reserve a non-empty test period")
    train_cycles = set(cycles[:train_count])
    validation_cycles = set(cycles[train_count : train_count + validation_count])
    test_cycles = set(cycles[train_count + validation_count :])
    if max(train_cycles) >= min(validation_cycles) or max(validation_cycles) >= min(test_cycles):
        raise FeatureError("Cycle ids are not in chronological order; use sortable cycle ids")
    train = ordered.loc[ordered["cycle_id"].isin(train_cycles)].reset_index(drop=True)
    validation = ordered.loc[ordered["cycle_id"].isin(validation_cycles)].reset_index(drop=True)
    test = ordered.loc[ordered["cycle_id"].isin(test_cycles)].reset_index(drop=True)
    logger.info(
        "Chronological split train=%d validation=%d test=%d cycles",
        len(train_cycles),
        len(validation_cycles),
        len(test_cycles),
    )
    return train, validation, test
