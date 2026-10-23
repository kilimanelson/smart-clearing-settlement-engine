"""Synthetic data stays reproducible and does not label itself from the future."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from smart_clearing.features import MODEL_FEATURES
from smart_clearing.simulator import (
    SNAPSHOT_COLUMNS,
    SimulatorConfig,
    build_snapshots,
    demo_config,
    label_shortfall,
    simulate,
)


def _stamp(hour: int, minute: int = 0, day: int = 2) -> datetime:
    return datetime(2024, 1, day, hour, minute, tzinfo=UTC)


def _transaction(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "transaction_id": "T1",
        "cycle_id": "CY-20240102",
        "issuer_bank": "A",
        "acquirer_bank": "B",
        "amount_minor": 10_000,
        "currency": "USD",
        "status": "cleared",
        "event_timestamp": _stamp(10),
        "clearing_timestamp": _stamp(10),
        "reversal_of_transaction_id": None,
    }
    row.update(overrides)
    return row


def _liquidity(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "cycle_id": "CY-20240102",
        "bank_id": "A",
        "prediction_timestamp": _stamp(14),
        "cutoff_timestamp": _stamp(17),
        "available_liquidity_at_prediction": 20_000,
        "available_liquidity_at_cutoff": 20_000,
        "scenario": "normal",
        "funding_shock_after_prediction": False,
    }
    row.update(overrides)
    return row


def _manual_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    transactions = pd.DataFrame(
        [
            _transaction(transaction_id="T1", amount_minor=10_000, event_timestamp=_stamp(10)),
            _transaction(transaction_id="T2", amount_minor=50_000, event_timestamp=_stamp(16)),
            _transaction(
                transaction_id="T3",
                amount_minor=9_999_999,
                currency="EUR",
                event_timestamp=_stamp(11),
            ),
            _transaction(
                transaction_id="T4",
                amount_minor=80_000,
                status="pending",
                event_timestamp=_stamp(9),
                clearing_timestamp=None,
            ),
            _transaction(
                transaction_id="T5",
                cycle_id="CY-20240103",
                amount_minor=1_000,
                event_timestamp=_stamp(10, day=3),
                clearing_timestamp=_stamp(10, day=3),
            ),
        ]
    )
    liquidity = pd.DataFrame(
        [
            _liquidity(bank_id="A"),
            _liquidity(
                bank_id="B",
                available_liquidity_at_prediction=5_000,
                available_liquidity_at_cutoff=5_000,
            ),
            _liquidity(
                cycle_id="CY-20240103",
                bank_id="A",
                prediction_timestamp=_stamp(14, day=3),
                cutoff_timestamp=_stamp(17, day=3),
                available_liquidity_at_prediction=5_000,
                available_liquidity_at_cutoff=0,
            ),
            _liquidity(
                cycle_id="CY-20240103",
                bank_id="B",
                prediction_timestamp=_stamp(14, day=3),
                cutoff_timestamp=_stamp(17, day=3),
                available_liquidity_at_prediction=5_000,
                available_liquidity_at_cutoff=5_000,
            ),
        ]
    )
    return transactions, liquidity


def test_shortfall_requires_the_debit_to_exceed_cutoff_liquidity() -> None:
    assert label_shortfall(-100, 100) == (100, 0)
    assert label_shortfall(-101, 100) == (101, 1)
    assert label_shortfall(50, 0) == (0, 0)


def test_snapshot_features_stop_at_the_prediction_time_and_labels_use_cutoff() -> None:
    transactions, liquidity = _manual_frames()
    snapshots = build_snapshots(transactions, liquidity, currency="USD")
    first = snapshots.loc[snapshots["cycle_id"] == "CY-20240102"].set_index("bank_id")
    second = snapshots.loc[snapshots["cycle_id"] == "CY-20240103"].set_index("bank_id")

    assert int(first.loc["A", "outgoing_volume_so_far"]) == 10_000
    assert int(first.loc["A", "transaction_count_so_far"]) == 2
    assert int(first.loc["A", "actual_final_net_debit"]) == 60_000
    assert int(first.loc["A", "actual_shortfall_at_cutoff"]) == 1
    assert int(first.loc["B", "actual_shortfall_at_cutoff"]) == 0
    assert float(first.loc["A", "historical_shortfall_rate"]) == 0.0
    assert float(second.loc["A", "historical_shortfall_rate"]) == 1.0
    assert float(second.loc["B", "historical_shortfall_rate"]) == 0.0
    assert (snapshots["prediction_timestamp"] < snapshots["cutoff_timestamp"]).all()
    assert list(snapshots.columns) == list(SNAPSHOT_COLUMNS)


def test_changing_cutoff_liquidity_changes_labels_only() -> None:
    """A cycle's own future liquidity changes its label, not its features.

    Historical shortfall rate may use earlier cycles that have already settled.
    This test therefore changes only the later cycle's cutoff liquidity.
    """
    transactions, liquidity = _manual_frames()
    baseline = build_snapshots(transactions, liquidity, currency="USD")
    richer = liquidity.copy()
    later = richer["cycle_id"] == "CY-20240103"
    richer.loc[later, "available_liquidity_at_cutoff"] = 10**12
    covered = build_snapshots(transactions, richer, currency="USD")
    pd.testing.assert_frame_equal(
        baseline.loc[:, list(MODEL_FEATURES)].reset_index(drop=True),
        covered.loc[:, list(MODEL_FEATURES)].reset_index(drop=True),
    )
    later_labels = baseline["cycle_id"] == "CY-20240103"
    assert int(baseline.loc[later_labels, "actual_shortfall_at_cutoff"].sum()) > 0
    assert int(covered.loc[later_labels, "actual_shortfall_at_cutoff"].sum()) == 0
    earlier = baseline["cycle_id"] == "CY-20240102"
    pd.testing.assert_series_equal(
        baseline.loc[earlier, "actual_shortfall_at_cutoff"].reset_index(drop=True),
        covered.loc[earlier, "actual_shortfall_at_cutoff"].reset_index(drop=True),
    )


def test_transactions_after_prediction_do_not_change_features() -> None:
    transactions, liquidity = _manual_frames()
    baseline = build_snapshots(transactions, liquidity, currency="USD")
    extra = pd.concat(
        [
            transactions,
            pd.DataFrame(
                [
                    _transaction(
                        transaction_id="FUTURE",
                        amount_minor=5_000_000,
                        event_timestamp=_stamp(16, 30),
                        clearing_timestamp=_stamp(16, 30),
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    updated = build_snapshots(extra, liquidity, currency="USD")
    pd.testing.assert_frame_equal(
        baseline.loc[:, list(MODEL_FEATURES)].reset_index(drop=True),
        updated.loc[:, list(MODEL_FEATURES)].reset_index(drop=True),
    )
    assert (
        not baseline["actual_final_net_debit"]
        .reset_index(drop=True)
        .equals(updated["actual_final_net_debit"].reset_index(drop=True))
    )


def test_same_seed_reproduces_the_dataset_and_another_seed_can_differ() -> None:
    config = SimulatorConfig(
        seed=7, n_cycles=4, n_banks=3, mean_transactions_per_cycle=12, min_transactions_per_cycle=8
    )
    first = simulate(config)
    second = simulate(config)
    other = simulate(
        SimulatorConfig(
            seed=8,
            n_cycles=4,
            n_banks=3,
            mean_transactions_per_cycle=12,
            min_transactions_per_cycle=8,
        )
    )
    pd.testing.assert_frame_equal(first.transactions, second.transactions)
    pd.testing.assert_frame_equal(first.snapshots, second.snapshots)
    assert not first.transactions["amount_minor"].equals(other.transactions["amount_minor"])


def test_documented_demo_covers_ids_time_bounds_statuses_and_both_labels() -> None:
    result = simulate(demo_config())
    transactions = result.transactions
    snapshots = result.snapshots
    assert transactions["transaction_id"].is_unique
    assert set(snapshots["actual_shortfall_at_cutoff"].unique()) == {0, 1}
    assert {"pending", "cleared", "reversed"}.issubset(set(transactions["status"].unique()))
    assert (snapshots["prediction_timestamp"] < snapshots["cutoff_timestamp"]).all()
    for cycle_id, group in transactions.groupby("cycle_id"):
        day = datetime.strptime(str(cycle_id), "CY-%Y%m%d").replace(tzinfo=UTC)
        assert group["event_timestamp"].min() >= pd.Timestamp(day)
        assert group["event_timestamp"].max() < pd.Timestamp(day + timedelta(days=1))
    assert snapshots["cycle_id"].nunique() == demo_config().n_cycles
    assert snapshots["bank_id"].nunique() == demo_config().n_banks
