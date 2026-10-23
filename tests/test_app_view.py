"""Dashboard view model: reconciliation, roles, and the non-execution warning."""

from datetime import UTC, datetime

import pandas as pd

from smart_clearing.app import build_cycle_view

_COLUMNS = [
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
]


def test_empty_cycle_has_an_empty_state() -> None:
    transactions = pd.DataFrame(columns=_COLUMNS)
    snapshots = pd.DataFrame(columns=["cycle_id", "bank_id"])
    view = build_cycle_view(
        transactions, snapshots, "C-EMPTY", metrics_error="Model evaluation was not run."
    )
    assert "no simulated transactions" in view.reconciliation_message.lower()
    assert view.metrics_error == "Model evaluation was not run."
    assert "not production-ready" in view.disclaimer


def test_view_reconciles_roles_and_does_not_execute_funding() -> None:
    event = datetime(2024, 1, 2, 10, tzinfo=UTC)
    transactions = pd.DataFrame(
        [
            {
                "transaction_id": "T1",
                "cycle_id": "C1",
                "issuer_bank": "A",
                "acquirer_bank": "B",
                "amount_minor": 1000,
                "currency": "USD",
                "status": "cleared",
                "event_timestamp": event,
                "clearing_timestamp": event,
                "reversal_of_transaction_id": None,
            }
        ]
    )
    snapshots = pd.DataFrame(
        [
            {
                "cycle_id": "C1",
                "bank_id": "A",
                "prediction_timestamp": datetime(2024, 1, 2, 14, tzinfo=UTC),
                "cutoff_timestamp": datetime(2024, 1, 2, 17, tzinfo=UTC),
                "predicted_or_forecast_net_debit": 1000,
                "available_liquidity_at_prediction": 100,
                "available_liquidity_at_cutoff": 100,
                "actual_shortfall_at_cutoff": 1,
                "actual_final_net_debit": 1000,
                "scenario": "normal",
            },
            {
                "cycle_id": "C1",
                "bank_id": "B",
                "prediction_timestamp": datetime(2024, 1, 2, 14, tzinfo=UTC),
                "cutoff_timestamp": datetime(2024, 1, 2, 17, tzinfo=UTC),
                "predicted_or_forecast_net_debit": 0,
                "available_liquidity_at_prediction": 5_000,
                "available_liquidity_at_cutoff": 5_000,
                "actual_shortfall_at_cutoff": 0,
                "actual_final_net_debit": 0,
                "scenario": "normal",
            },
        ]
    )
    scores = pd.DataFrame(
        [
            {"cycle_id": "C1", "bank_id": "A", "probability": 0.9, "predicted_label": 1},
            {"cycle_id": "C1", "bank_id": "B", "probability": 0.1, "predicted_label": 0},
        ]
    )
    view = build_cycle_view(transactions, snapshots, "C1", scores=scores, funding_limit_minor=100)
    assert view.reconciled is True
    roles = {row["bank_id"]: row["role"] for row in view.positions}
    assert roles == {"A": "Net debtor", "B": "Net creditor"}
    assert all("USD" in row["net_position"] for row in view.positions)
    assert view.probabilities[0]["simulated_outcome"] in {"Shortfall", "Sufficient"}
    assert view.probabilities[0]["model_probability"] != view.probabilities[0]["simulated_outcome"]
    assert view.unresolved
    assert all(row["executed"] == "No — recommendation only" for row in view.funding_rows)
