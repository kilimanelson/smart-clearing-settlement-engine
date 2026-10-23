"""Synthetic card-payment activity for tests and model training.

Every figure this module emits is simulated. Bank volumes, liquidity, and
shortfalls are assumptions, not observations of real institutions.

Two datasets are produced:

* Transactions, one row per card event.
* Bank-cycle snapshots. Feature columns use only events at or before the
  prediction timestamp. ``actual_shortfall_at_cutoff`` is computed later from
  the cutoff-time net debit and cutoff-time liquidity.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import numpy as np
import pandas as pd

from smart_clearing.clearing import clear_cycle
from smart_clearing.features import compute_observed_features
from smart_clearing.schemas import Transaction, TransactionStatus, ensure_utc

logger = logging.getLogger(__name__)

TRANSACTION_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "cycle_id",
    "issuer_bank",
    "acquirer_bank",
    "amount_minor",
    "currency",
    "status",
    "event_timestamp",
    "clearing_timestamp",
    "reversal_of_transaction_id",
)

SNAPSHOT_COLUMNS: tuple[str, ...] = (
    "cycle_id",
    "bank_id",
    "prediction_timestamp",
    "cutoff_timestamp",
    "incoming_volume_so_far",
    "outgoing_volume_so_far",
    "transaction_count_so_far",
    "average_transaction_amount_so_far",
    "recent_volume_change",
    "historical_shortfall_rate",
    "available_liquidity_at_prediction",
    "predicted_or_forecast_net_debit",
    "actual_shortfall_at_cutoff",
    "actual_final_net_debit",
    "available_liquidity_at_cutoff",
    "scenario",
    "funding_shock_after_prediction",
)

_ISSUER_PROFILE = (0.34, 0.24, 0.18, 0.14, 0.10, 0.07, 0.05, 0.04)
_ACQUIRER_PROFILE = (0.05, 0.08, 0.11, 0.13, 0.16, 0.18, 0.14, 0.15)
_LIQUIDITY_PROFILE = (0.40, 0.62, 0.95, 1.30, 1.75, 2.40, 2.80, 3.20)
_DOW_MULTIPLIER = {0: 1.12, 1: 1.05, 2: 1.00, 3: 1.00, 4: 0.88}


def synthetic_assumptions() -> list[str]:
    """Assumptions that bound every dataset this module generates."""
    return [
        "Participants are fictional bank ids. Nothing here is a real institution's flow.",
        "Each cycle is one weekday. The simulation clock is UTC, and the intraday curve uses that clock.",
        "A purchase creates an issuer-to-acquirer obligation. Issuer and acquirer are always different.",
        "Amounts are integer minor units drawn from a lognormal, then clipped.",
        "Pending records stay pending. They are visible in the transaction file and ineligible for DNS.",
        "A reversal is a separate record that references the original. The original purchase remains cleared.",
        "One settlement currency is cleared per cycle. A minority of records are tagged with a second currency and then excluded.",
        "Prediction time is 14:00 and cutoff is 17:00. Features cannot see later events.",
        "The intraday forecast scales observed eligible volume by a fixed prior. It is not fit on that cycle's future.",
        "Liquidity at prediction time is the expected lognormal amount times the mean transaction count times that bank's issuer weight, times a fixed liquidity factor and small noise. It does not use the realized transactions.",
        "A funding shock can reduce liquidity after prediction time and before cutoff. That shocked balance is a label input, not a feature.",
        "A shortfall means the eligible net debit at cutoff is strictly greater than liquidity at cutoff.",
        "Historical shortfall rate uses only earlier cycles. The current label is applied after the snapshot is built.",
        "Net debit of zero (a creditor or a flat bank) is not a shortfall.",
        "These settings are chosen so a demonstration run contains both shortfalls and non-shortfalls. They are not a calibrated model of a real market.",
    ]


@dataclass(frozen=True)
class SimulatorConfig:
    """Controls one reproducible simulation run."""

    seed: int = 42
    n_banks: int = 4
    n_cycles: int = 18
    currency: str = "USD"
    foreign_currency: str = "EUR"
    foreign_currency_rate: float = 0.03
    start_date: date = date(2024, 1, 2)
    mean_transactions_per_cycle: int = 28
    min_transactions_per_cycle: int = 12
    pending_rate: float = 0.08
    reversal_rate: float = 0.06
    volume_shock_probability: float = 0.18
    funding_shock_probability: float = 0.28
    amount_log_mean: float = 8.5
    amount_log_sigma: float = 0.8
    min_amount_minor: int = 100
    max_amount_minor: int = 2_500_000
    prediction_hour: int = 14
    cutoff_hour: int = 17

    def __post_init__(self) -> None:
        if not 2 <= self.n_banks <= 8:
            raise ValueError("n_banks must be between 2 and 8")
        if self.n_cycles < 1:
            raise ValueError("n_cycles must be positive")
        if not (0 <= self.prediction_hour < self.cutoff_hour <= 23):
            raise ValueError("prediction_hour must be earlier than cutoff_hour")
        if self.min_transactions_per_cycle < 1:
            raise ValueError("min_transactions_per_cycle must be positive")
        if self.foreign_currency == self.currency:
            raise ValueError("foreign_currency must differ from the settlement currency")


def demo_config(seed: int = 42, n_cycles: int = 18) -> SimulatorConfig:
    """Small configuration shared by the dashboard and the automated tests."""
    return SimulatorConfig(seed=seed, n_cycles=n_cycles, n_banks=4, mean_transactions_per_cycle=28)


def experiment_config(seed: int = 42) -> SimulatorConfig:
    """Larger configuration used for the written evaluation report."""
    return SimulatorConfig(
        seed=seed,
        n_cycles=36,
        n_banks=6,
        mean_transactions_per_cycle=48,
        min_transactions_per_cycle=20,
    )


@dataclass(frozen=True)
class SimulationResult:
    """Transactions, the liquidity schedule, and the labeled snapshot table."""

    transactions: pd.DataFrame
    liquidity: pd.DataFrame
    snapshots: pd.DataFrame
    config: SimulatorConfig


def label_shortfall(net_position_minor: int, available_liquidity_at_cutoff: int) -> tuple[int, int]:
    """Return ``(net_debit_minor, shortfall_label)`` from a cleared position.

    Net debit is the positive amount owed. Liquidity that merely equals the
    debit is sufficient, because a shortfall requires the obligation to exceed
    available liquidity.
    """
    if isinstance(net_position_minor, bool) or not isinstance(net_position_minor, int):
        raise TypeError("net_position_minor must be an int")
    if isinstance(available_liquidity_at_cutoff, bool) or not isinstance(
        available_liquidity_at_cutoff, int
    ):
        raise TypeError("available_liquidity_at_cutoff must be an int")
    if available_liquidity_at_cutoff < 0:
        raise ValueError("available liquidity cannot be negative")
    net_debit = -net_position_minor if net_position_minor < 0 else 0
    shortfall = 1 if net_debit > available_liquidity_at_cutoff else 0
    return net_debit, shortfall


def _business_days(start: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _normalize(profile: Sequence[float], size: int) -> np.ndarray:
    raw = np.asarray(profile[:size], dtype=float)
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("activity profile must be positive")
    return raw / total


def _as_int(value: object) -> int:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("integer field received a bool or float")
    return int(value)  # type: ignore[arg-type]


def _optional_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return _as_timestamp(value)


def _as_timestamp(value: object) -> datetime:
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise TypeError("expected a datetime")
    return ensure_utc(value)


def _optional_id(value: object) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    if text.lower() == "nan" or text == "":
        return None
    return text


def frame_to_transactions(frame: pd.DataFrame) -> list[Transaction]:
    """Convert a transaction table into validated records."""
    records: list[Transaction] = []
    for row in frame.to_dict(orient="records"):
        status = row["status"]
        if not isinstance(status, TransactionStatus):
            status = TransactionStatus(str(status))
        records.append(
            Transaction(
                transaction_id=str(row["transaction_id"]),
                cycle_id=str(row["cycle_id"]),
                issuer_bank=str(row["issuer_bank"]),
                acquirer_bank=str(row["acquirer_bank"]),
                amount_minor=_as_int(row["amount_minor"]),
                currency=str(row["currency"]),
                status=status,
                event_timestamp=_as_timestamp(row["event_timestamp"]),
                clearing_timestamp=_optional_timestamp(row["clearing_timestamp"]),
                reversal_of_transaction_id=_optional_id(row["reversal_of_transaction_id"]),
            )
        )
    return records


def _choose_indexes(rng: np.random.Generator, pool: np.ndarray, size: int) -> set[int]:
    if size <= 0 or len(pool) == 0:
        return set()
    chosen = rng.choice(pool, size=min(size, len(pool)), replace=False)
    return {int(index) for index in np.atleast_1d(chosen)}


def _generate_activity(
    config: SimulatorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.seed)
    banks = [f"BANK_{index:02d}" for index in range(config.n_banks)]
    issuer_weights = _normalize(_ISSUER_PROFILE, config.n_banks)
    acquirer_weights = _normalize(_ACQUIRER_PROFILE, config.n_banks)
    liquidity_factors = np.asarray(_LIQUIDITY_PROFILE[: config.n_banks], dtype=float)
    transactions: list[dict[str, Any]] = []
    liquidity_rows: list[dict[str, Any]] = []

    for cycle_date in _business_days(config.start_date, config.n_cycles):
        cycle_id = f"CY-{cycle_date:%Y%m%d}"
        cycle_start = datetime.combine(cycle_date, time.min, tzinfo=UTC)
        prediction_at = cycle_start.replace(hour=config.prediction_hour)
        cutoff_at = cycle_start.replace(hour=config.cutoff_hour)
        multiplier = 1.0
        if float(rng.random()) < config.volume_shock_probability:
            multiplier = float(rng.choice(np.array([0.45, 1.85, 2.4])))
        mean_count = (
            config.mean_transactions_per_cycle * _DOW_MULTIPLIER[cycle_date.weekday()] * multiplier
        )
        count = max(config.min_transactions_per_cycle, int(rng.poisson(mean_count)))
        count = min(count, config.mean_transactions_per_cycle * 5)
        drafts: list[dict[str, Any]] = []
        for sequence in range(count):
            issuer_index = int(rng.choice(config.n_banks, p=issuer_weights))
            acquirer_index = int(rng.choice(config.n_banks, p=acquirer_weights))
            while acquirer_index == issuer_index:
                acquirer_index = int(rng.choice(config.n_banks, p=acquirer_weights))
            raw_amount = float(rng.lognormal(config.amount_log_mean, config.amount_log_sigma))
            amount = int(
                round(min(config.max_amount_minor, max(config.min_amount_minor, raw_amount)))
            )
            if float(rng.random()) < 0.30:
                minute = int(
                    rng.integers(config.prediction_hour * 60, config.cutoff_hour * 60 + 45)
                )
            else:
                minute = int(rng.normal(11.5 * 60, 90))
                minute = int(np.clip(minute, 8 * 60, config.cutoff_hour * 60 - 1))
            event_at = cycle_start + timedelta(minutes=minute)
            drafts.append(
                {
                    "transaction_id": f"TX-{cycle_id}-{sequence:05d}",
                    "cycle_id": cycle_id,
                    "issuer_bank": banks[issuer_index],
                    "acquirer_bank": banks[acquirer_index],
                    "amount_minor": amount,
                    "currency": config.currency,
                    "status": TransactionStatus.CLEARED.value,
                    "event_timestamp": event_at,
                    "clearing_timestamp": event_at,
                    "reversal_of_transaction_id": None,
                }
            )
        pool = np.arange(count)
        pending_n = min(int(round(count * config.pending_rate)), count)
        pending_indexes = _choose_indexes(rng, pool, pending_n)
        remaining = np.array([index for index in pool if int(index) not in pending_indexes])
        reversal_n = min(int(round(count * config.reversal_rate)), len(remaining))
        reversal_indexes = _choose_indexes(rng, remaining, reversal_n)
        foreign_pool = np.array(
            [index for index in remaining if int(index) not in reversal_indexes],
            dtype=int,
        )
        foreign_n = min(int(round(count * config.foreign_currency_rate)), len(foreign_pool))
        foreign_indexes = _choose_indexes(rng, foreign_pool, foreign_n)
        next_sequence = count
        for index, draft in enumerate(drafts):
            if index in pending_indexes:
                draft["status"] = TransactionStatus.PENDING.value
                draft["clearing_timestamp"] = None
            elif index in foreign_indexes:
                draft["currency"] = config.foreign_currency
            if index in reversal_indexes:
                delay = int(rng.integers(5, 90))
                reversal_at = min(
                    draft["event_timestamp"] + timedelta(minutes=delay),
                    cycle_start + timedelta(hours=23, minutes=59),
                )
                transactions.append(draft)
                transactions.append(
                    {
                        "transaction_id": f"TX-{cycle_id}-{next_sequence:05d}",
                        "cycle_id": cycle_id,
                        "issuer_bank": draft["issuer_bank"],
                        "acquirer_bank": draft["acquirer_bank"],
                        "amount_minor": draft["amount_minor"],
                        "currency": draft["currency"],
                        "status": TransactionStatus.REVERSED.value,
                        "event_timestamp": reversal_at,
                        "clearing_timestamp": reversal_at,
                        "reversal_of_transaction_id": draft["transaction_id"],
                    }
                )
                next_sequence += 1
            else:
                transactions.append(draft)

        shocked_any = False
        typical_amount = math.exp(config.amount_log_mean + 0.5 * config.amount_log_sigma**2)
        typical_amount = min(
            float(config.max_amount_minor), max(float(config.min_amount_minor), typical_amount)
        )
        for bank_index, bank_id in enumerate(banks):
            expected_outgoing = (
                typical_amount
                * config.mean_transactions_per_cycle
                * float(issuer_weights[bank_index])
            )
            noise = float(rng.uniform(0.9, 1.1))
            liquidity = max(
                0, int(expected_outgoing * float(liquidity_factors[bank_index]) * noise)
            )
            shocked = bool(float(rng.random()) < config.funding_shock_probability)
            shocked_any = shocked_any or shocked
            cutoff_liquidity = (
                int(liquidity * float(rng.uniform(0.25, 0.55))) if shocked else liquidity
            )
            liquidity_rows.append(
                {
                    "cycle_id": cycle_id,
                    "bank_id": bank_id,
                    "prediction_timestamp": prediction_at,
                    "cutoff_timestamp": cutoff_at,
                    "available_liquidity_at_prediction": liquidity,
                    "available_liquidity_at_cutoff": cutoff_liquidity,
                    "funding_shock_after_prediction": shocked,
                    "scenario": "pending",
                }
            )
        if multiplier >= 1.5:
            scenario = "volume_spike"
        elif multiplier <= 0.6:
            scenario = "volume_drop"
        elif shocked_any:
            scenario = "funding_shock"
        else:
            scenario = "normal"
        for row in liquidity_rows:
            if row["cycle_id"] == cycle_id:
                row["scenario"] = scenario

    transaction_frame = pd.DataFrame(transactions, columns=list(TRANSACTION_COLUMNS))
    liquidity_frame = pd.DataFrame(liquidity_rows)
    transaction_frame["event_timestamp"] = pd.to_datetime(
        transaction_frame["event_timestamp"], utc=True
    )
    transaction_frame["clearing_timestamp"] = pd.to_datetime(
        transaction_frame["clearing_timestamp"], utc=True
    )
    for column in ("prediction_timestamp", "cutoff_timestamp"):
        liquidity_frame[column] = pd.to_datetime(liquidity_frame[column], utc=True)
    transaction_frame = transaction_frame.sort_values(
        ["cycle_id", "event_timestamp", "transaction_id"]
    ).reset_index(drop=True)
    liquidity_frame = liquidity_frame.sort_values(["cycle_id", "bank_id"]).reset_index(drop=True)
    logger.info(
        "Simulated %d transactions across %d cycles and %d banks",
        len(transaction_frame),
        liquidity_frame["cycle_id"].nunique(),
        config.n_banks,
    )
    return transaction_frame, liquidity_frame


def build_snapshots(
    transactions: pd.DataFrame,
    liquidity: pd.DataFrame,
    *,
    currency: str,
) -> pd.DataFrame:
    """Label each bank-cycle using only the information permitted at each time.

    Historical shortfall rates are updated after the current label is stored,
    so the current outcome cannot flow back into its own features.
    """
    required = {
        "cycle_id",
        "bank_id",
        "prediction_timestamp",
        "cutoff_timestamp",
        "available_liquidity_at_prediction",
        "available_liquidity_at_cutoff",
        "scenario",
        "funding_shock_after_prediction",
    }
    missing = required.difference(liquidity.columns)
    if missing:
        raise ValueError(f"liquidity schedule is missing columns: {sorted(missing)}")
    schedule = liquidity.sort_values(["prediction_timestamp", "bank_id"]).reset_index(drop=True)
    history: dict[str, list[int]] = {}
    rows: list[dict[str, Any]] = []
    cycle_ids = list(dict.fromkeys(schedule["cycle_id"].tolist()))
    for cycle_id in cycle_ids:
        cycle_liquidity = schedule.loc[schedule["cycle_id"] == cycle_id]
        prediction_at = _as_timestamp(cycle_liquidity["prediction_timestamp"].iloc[0])
        cutoff_at = _as_timestamp(cycle_liquidity["cutoff_timestamp"].iloc[0])
        if prediction_at >= cutoff_at:
            raise ValueError(f"{cycle_id} prediction timestamp must precede cutoff")
        banks = sorted(cycle_liquidity["bank_id"].unique())
        cycle_transactions = transactions.loc[
            (transactions["cycle_id"] == cycle_id) & (transactions["currency"] == currency)
        ].copy()
        cycle_transactions["event_timestamp"] = pd.to_datetime(
            cycle_transactions["event_timestamp"], utc=True
        )
        observed = cycle_transactions.loc[cycle_transactions["event_timestamp"] <= prediction_at]
        settled = cycle_transactions.loc[cycle_transactions["event_timestamp"] <= cutoff_at]
        rates = {
            bank_id: (sum(history.get(bank_id, [])) / len(history[bank_id]))
            if history.get(bank_id)
            else 0.0
            for bank_id in banks
        }
        liquidity_map = {
            str(row.bank_id): int(row.available_liquidity_at_prediction)
            for row in cycle_liquidity.itertuples(index=False)
        }
        features = compute_observed_features(
            frame_to_transactions(observed),
            cycle_id=str(cycle_id),
            currency=currency,
            participants=banks,
            prediction_timestamp=prediction_at,
            historical_rates=rates,
            liquidity_at_prediction=liquidity_map,
        )
        cutoff_report = clear_cycle(
            frame_to_transactions(settled),
            cycle_id=str(cycle_id),
            currency=currency,
            participants=banks,
        )
        positions = {position.bank_id: position for position in cutoff_report.net_positions}
        feature_by_bank = {str(row["bank_id"]): row for row in features}
        for bank_row in cycle_liquidity.itertuples(index=False):
            bank_id = str(bank_row.bank_id)
            net_debit, shortfall = label_shortfall(
                positions[bank_id].net_position_minor,
                int(bank_row.available_liquidity_at_cutoff),
            )
            feature_row = feature_by_bank[bank_id]
            rows.append(
                {
                    "cycle_id": str(cycle_id),
                    "bank_id": bank_id,
                    "prediction_timestamp": prediction_at,
                    "cutoff_timestamp": cutoff_at,
                    "incoming_volume_so_far": feature_row["incoming_volume_so_far"],
                    "outgoing_volume_so_far": feature_row["outgoing_volume_so_far"],
                    "transaction_count_so_far": feature_row["transaction_count_so_far"],
                    "average_transaction_amount_so_far": feature_row[
                        "average_transaction_amount_so_far"
                    ],
                    "recent_volume_change": feature_row["recent_volume_change"],
                    "historical_shortfall_rate": feature_row["historical_shortfall_rate"],
                    "available_liquidity_at_prediction": feature_row[
                        "available_liquidity_at_prediction"
                    ],
                    "predicted_or_forecast_net_debit": feature_row[
                        "predicted_or_forecast_net_debit"
                    ],
                    "actual_shortfall_at_cutoff": shortfall,
                    "actual_final_net_debit": net_debit,
                    "available_liquidity_at_cutoff": int(bank_row.available_liquidity_at_cutoff),
                    "scenario": str(bank_row.scenario),
                    "funding_shock_after_prediction": bool(bank_row.funding_shock_after_prediction),
                }
            )
            history.setdefault(bank_id, []).append(shortfall)
    snapshot = pd.DataFrame(rows, columns=list(SNAPSHOT_COLUMNS))
    return snapshot.sort_values(["cycle_id", "bank_id"]).reset_index(drop=True)


def simulate(config: SimulatorConfig | None = None) -> SimulationResult:
    """Generate transactions and labeled snapshots from ``config``."""
    resolved = config or demo_config()
    transactions, liquidity = _generate_activity(resolved)
    snapshots = build_snapshots(transactions, liquidity, currency=resolved.currency)
    return SimulationResult(
        transactions=transactions,
        liquidity=liquidity,
        snapshots=snapshots,
        config=resolved,
    )
