"""Streamlit operations view over the synthetic clearing simulation.

Financial calculations live in the clearing, model, and optimizer modules.
This module only chooses a cycle and renders those results. No control here
submits a payment or changes a DNS obligation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Community Cloud runs this file directly and does not install the local
# package. The script directory is then ``src/smart_clearing``, so the package
# parent has to be importable before the imports below.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smart_clearing.clearing import ClearingError, clear_cycle
from smart_clearing.config import SIMULATION_DISCLAIMER, configure_logging
from smart_clearing.model import InsufficientDataError, score_snapshots, train_and_evaluate
from smart_clearing.optimizer import (
    ActualLiquidityOutcome,
    compare_funding_strategies,
    evaluate_strategy,
)
from smart_clearing.schemas import (
    FundingStrategy,
    LiquidityInput,
    OptimizerConfig,
    format_minor_units,
)
from smart_clearing.simulator import demo_config, frame_to_transactions, simulate

_DEBTOR = "#9b3a3a"
_CREDITOR = "#1f7a4d"
_FLAT = "#5c6b7a"


@dataclass
class CycleView:
    """Presentation-ready facts for one simulated cycle."""

    disclaimer: str
    cycle_id: str
    currency: str
    error: str | None = None
    reconciled: bool = False
    reconciliation_message: str = ""
    transaction_count: int = 0
    late_count: int = 0
    eligible_amount_minor: int = 0
    purchase_count: int = 0
    reversal_count: int = 0
    quarantined_count: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    positions: list[dict[str, Any]] = field(default_factory=list)
    probabilities: list[dict[str, Any]] = field(default_factory=list)
    funding_rows: list[dict[str, Any]] = field(default_factory=list)
    strategy_rows: list[dict[str, Any]] = field(default_factory=list)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    metrics_rows: list[dict[str, Any]] = field(default_factory=list)
    metrics_error: str | None = None
    selected_model: str | None = None
    model_threshold: float | None = None
    display_threshold: float = 0.5
    split_name: str | None = None
    limitation: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    transactions: list[dict[str, Any]] = field(default_factory=list)


def _money(amount: int, currency: str) -> str:
    return format_minor_units(int(amount), currency)


def _role(net_position: int) -> str:
    if net_position < 0:
        return "Net debtor"
    if net_position > 0:
        return "Net creditor"
    return "Flat"


def _history(snapshots: pd.DataFrame, currency: str) -> list[dict[str, Any]]:
    if snapshots.empty:
        return []
    rows: list[dict[str, Any]] = []
    for cycle_id, group in snapshots.groupby("cycle_id", sort=True):
        rows.append(
            {
                "cycle_id": str(cycle_id),
                "scenario": str(group["scenario"].iloc[0]),
                "banks": int(len(group)),
                "simulated_shortfalls": int(group["actual_shortfall_at_cutoff"].sum()),
                "simulated_net_debit": _money(int(group["actual_final_net_debit"].sum()), currency),
            }
        )
    return rows


def build_cycle_view(
    transactions: pd.DataFrame,
    snapshots: pd.DataFrame,
    cycle_id: str,
    *,
    scores: pd.DataFrame | None = None,
    metrics: dict[str, Any] | None = None,
    metrics_error: str | None = None,
    display_threshold: float = 0.5,
    funding_limit_minor: int = 20_000_000,
    liquidity_cost_bps: int = 25,
    risk_limit: float = 0.2,
    portfolio_cap_minor: int | None = None,
    fixed_buffer_percent: int = 10,
) -> CycleView:
    """Assemble one cycle from the clearing engine, scores, and optimizer."""
    currency = "USD"
    cycle_transactions = transactions.loc[transactions["cycle_id"] == cycle_id].copy()
    if not cycle_transactions.empty:
        currency = str(cycle_transactions["currency"].mode().iloc[0])
    cycle_snapshots = snapshots.loc[snapshots["cycle_id"] == cycle_id].copy()
    view = CycleView(
        disclaimer=SIMULATION_DISCLAIMER,
        cycle_id=cycle_id,
        currency=currency,
        metrics_error=metrics_error,
        display_threshold=display_threshold,
        history=_history(snapshots, currency),
    )
    if metrics is not None:
        view.selected_model = str(metrics.get("selected_model"))
        view.limitation = str(metrics.get("limitation")) if metrics.get("limitation") else None
        view.model_threshold = _threshold_for(metrics, view.selected_model)
        view.split_name = _split_name(metrics, cycle_id)
        metric_order = (
            "name",
            "test_f1",
            "test_precision",
            "test_recall",
            "test_average_precision",
            "test_brier",
            "false_positives",
            "false_negatives",
            "true_positives",
            "true_negatives",
            "threshold",
            "validation_f1",
            "calibrated",
            "explanation",
        )
        view.metrics_rows = [
            {
                key: round(run[key], 3) if isinstance(run[key], float) else run[key]
                for key in metric_order
                if key in run
            }
            for run in metrics.get("runs", [])
        ]
    if cycle_transactions.empty and cycle_snapshots.empty:
        view.reconciliation_message = "This cycle has no simulated transactions."
        return view

    settlement_input = cycle_transactions
    late = cycle_transactions.iloc[0:0]
    if not cycle_snapshots.empty:
        cutoff = pd.Timestamp(cycle_snapshots["cutoff_timestamp"].iloc[0])
        event_times = pd.to_datetime(cycle_transactions["event_timestamp"], utc=True)
        settlement_input = cycle_transactions.loc[event_times <= cutoff]
        late = cycle_transactions.loc[event_times > cutoff]
    view.late_count = int(len(late))
    view.transaction_count = int(len(cycle_transactions))
    view.status_counts = {
        str(status): int(count)
        for status, count in cycle_transactions["status"].value_counts().sort_index().items()
    }
    participants = sorted(
        set(cycle_transactions["issuer_bank"])
        .union(cycle_transactions["acquirer_bank"])
        .union(cycle_snapshots["bank_id"] if not cycle_snapshots.empty else [])
    )
    try:
        report = clear_cycle(
            frame_to_transactions(settlement_input),
            cycle_id=cycle_id,
            currency=currency,
            participants=participants,
        )
    except ClearingError as exc:
        view.error = str(exc)
        view.reconciled = False
        view.reconciliation_message = "Clearing failed. No settlement report is available."
        return view

    view.reconciled = report.reconciled and report.net_position_sum_minor == 0
    view.reconciliation_message = (
        f"Reconciled. Net positions sum to {report.net_position_sum_minor} minor units."
        if view.reconciled
        else "This cycle did not reconcile."
    )
    view.purchase_count = report.purchase_count
    view.reversal_count = report.reversal_count
    view.quarantined_count = report.quarantined_count
    view.eligible_amount_minor = sum(item.amount_minor for item in report.bilateral_obligations)
    quarantine = {item.transaction_id: item.reason_code.value for item in report.quarantined}
    late_ids = set(late["transaction_id"].astype(str)) if not late.empty else set()
    for record in frame_to_transactions(cycle_transactions):
        if record.transaction_id in late_ids:
            treatment = "After cutoff, excluded from settlement"
        elif record.transaction_id in quarantine:
            treatment = f"Quarantined: {quarantine[record.transaction_id]}"
        else:
            treatment = "Included in settlement input"
        view.transactions.append(
            {
                "transaction_id": record.transaction_id,
                "status": record.status.value,
                "issuer_bank": record.issuer_bank,
                "acquirer_bank": record.acquirer_bank,
                "amount": _money(record.amount_minor, record.currency),
                "currency": record.currency,
                "event_timestamp": record.event_timestamp.isoformat(),
                "treatment": treatment,
            }
        )
    for position in report.net_positions:
        view.positions.append(
            {
                "bank_id": position.bank_id,
                "role": _role(position.net_position_minor),
                "net_position_minor": position.net_position_minor,
                "net_position": _money(position.net_position_minor, currency),
                "gross_debit": _money(position.gross_debit_minor, currency),
                "gross_credit": _money(position.gross_credit_minor, currency),
                "color": _DEBTOR
                if position.net_position_minor < 0
                else _CREDITOR
                if position.net_position_minor > 0
                else _FLAT,
            }
        )
    _attach_risk_and_funding(
        view,
        cycle_snapshots,
        scores,
        funding_limit_minor=funding_limit_minor,
        liquidity_cost_bps=liquidity_cost_bps,
        risk_limit=risk_limit,
        portfolio_cap_minor=portfolio_cap_minor,
        fixed_buffer_percent=fixed_buffer_percent,
    )
    return view


def _attach_risk_and_funding(
    view: CycleView,
    cycle_snapshots: pd.DataFrame,
    scores: pd.DataFrame | None,
    *,
    funding_limit_minor: int,
    liquidity_cost_bps: int,
    risk_limit: float,
    portfolio_cap_minor: int | None,
    fixed_buffer_percent: int,
) -> None:
    score_by_bank: dict[str, Any] = {}
    if scores is not None and not scores.empty:
        selected = scores.loc[scores["cycle_id"] == view.cycle_id]
        score_by_bank = {str(row.bank_id): row for row in selected.itertuples(index=False)}
    inputs: list[LiquidityInput] = []
    actuals: list[ActualLiquidityOutcome] = []
    threshold = view.model_threshold if view.model_threshold is not None else view.display_threshold
    for row in cycle_snapshots.itertuples(index=False):
        bank_id = str(row.bank_id)
        score = score_by_bank.get(bank_id)
        probability = None if score is None else float(score.probability)
        if probability is None:
            model_decision = "No score"
            flagged = False
        else:
            model_decision = (
                "Predicted shortfall" if probability >= threshold else "Predicted sufficient"
            )
            flagged = probability >= view.display_threshold
        outcome = int(row.actual_shortfall_at_cutoff)
        view.probabilities.append(
            {
                "bank_id": bank_id,
                "model_probability": None if probability is None else round(probability, 4),
                "simulated_outcome": "Shortfall" if outcome == 1 else "Sufficient",
                "model_decision": model_decision,
                "flagged_at_display_threshold": flagged,
                "forecast_net_debit": _money(
                    int(row.predicted_or_forecast_net_debit), view.currency
                ),
                "liquidity_at_prediction": _money(
                    int(row.available_liquidity_at_prediction), view.currency
                ),
                "liquidity_at_cutoff": _money(
                    int(row.available_liquidity_at_cutoff), view.currency
                ),
            }
        )
        if probability is None:
            continue
        inputs.append(
            LiquidityInput(
                bank_id=bank_id,
                shortfall_probability=probability,
                forecast_net_debit_minor=int(row.predicted_or_forecast_net_debit),
                available_liquidity_minor=int(row.available_liquidity_at_prediction),
                liquidity_cost_bps=liquidity_cost_bps,
                funding_limit_minor=funding_limit_minor,
            )
        )
        actuals.append(
            ActualLiquidityOutcome(
                bank_id=bank_id,
                actual_net_debit_minor=int(row.actual_final_net_debit),
                liquidity_at_cutoff_minor=int(row.available_liquidity_at_cutoff),
            )
        )
    if not inputs:
        return
    config = OptimizerConfig(
        risk_limit=risk_limit,
        fixed_buffer_percent=fixed_buffer_percent,
        portfolio_funding_cap_minor=portfolio_cap_minor,
    )
    for result in compare_funding_strategies(inputs, config):
        evaluation = evaluate_strategy(result, actuals)
        view.strategy_rows.append(
            {
                "strategy": result.strategy.value,
                "incremental_funding": _money(result.total_funding_minor, view.currency),
                "simulated_funding_cost": _money(result.total_cost_minor, view.currency),
                "simulated_shortfalls_after_funding": evaluation.shortfall_count,
                "unresolved_banks": result.unresolved_count,
                "simulated_shortfall_rate": round(evaluation.shortfall_rate, 3),
                "mean_utilization": None
                if evaluation.mean_utilization is None
                else round(evaluation.mean_utilization, 3),
            }
        )
        if result.strategy != FundingStrategy.RISK_INFORMED:
            continue
        for recommendation in result.recommendations:
            view.funding_rows.append(
                {
                    "bank_id": recommendation.bank_id,
                    "additional_funding": _money(
                        recommendation.additional_funding_minor, view.currency
                    ),
                    "executed": "No — recommendation only",
                    "unresolved_risk": recommendation.unresolved_risk,
                    "requirement": _money(recommendation.requirement_minor, view.currency),
                    "simulated_cost": _money(recommendation.incremental_cost_minor, view.currency),
                    "residual_gap": _money(recommendation.residual_gap_minor, view.currency),
                }
            )
            if recommendation.unresolved_risk:
                view.unresolved.append(
                    {
                        "bank_id": recommendation.bank_id,
                        "residual_gap": _money(recommendation.residual_gap_minor, view.currency),
                    }
                )


def _threshold_for(metrics: dict[str, Any], selected: str | None) -> float | None:
    for run in metrics.get("runs", []):
        if run.get("name") == selected:
            return float(run["threshold"])
    return None


def _split_name(metrics: dict[str, Any], cycle_id: str) -> str | None:
    for name in ("train", "validation", "test"):
        if cycle_id in set(metrics.get(f"{name}_cycles", [])):
            return name
    return None


def comparison_to_dict(report: Any) -> dict[str, Any]:
    """Convert a comparison report into plain values Streamlit can cache."""
    return {
        "selected_model": report.selected_model,
        "limitation": report.limitation,
        "train_positive_rate": report.train_positive_rate,
        "test_positive_rate": report.test_positive_rate,
        "train_cycles": list(report.train_cycles),
        "validation_cycles": list(report.validation_cycles),
        "test_cycles": list(report.test_cycles),
        "runs": [
            {
                "name": run.name,
                "threshold": run.threshold,
                "calibrated": run.calibrated,
                "explanation": run.explanation,
                "validation_f1": run.validation.f1,
                "test_f1": run.test.f1,
                "test_precision": run.test.precision,
                "test_recall": run.test.recall,
                "test_brier": run.test.brier,
                "test_average_precision": run.test.average_precision,
                "false_positives": run.test.false_positive,
                "false_negatives": run.test.false_negative,
                "true_positives": run.test.true_positive,
                "true_negatives": run.test.true_negative,
            }
            for run in report.runs
        ],
    }


@st.cache_data(show_spinner="Simulating settlement cycles...")
def cached_simulation(seed: int, n_cycles: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the demo datasets. Results are identical for the same seed."""
    result = simulate(demo_config(seed=seed, n_cycles=n_cycles))
    return result.transactions, result.snapshots


@st.cache_data(show_spinner="Fitting shortfall models on the simulated cycles...")
def cached_metrics(seed: int, n_cycles: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the three approaches and score every snapshot with the selected model."""
    _transactions, snapshots = cached_simulation(seed, n_cycles)
    trained = train_and_evaluate(snapshots, seed=seed)
    scores = score_snapshots(trained.model, snapshots)
    return scores, comparison_to_dict(trained.report)


def position_figure(view: CycleView) -> go.Figure:
    """Bar chart of signed net positions. Display scale only; values stay in minor units upstream."""
    figure = go.Figure(
        go.Bar(
            x=[row["bank_id"] for row in view.positions],
            y=[row["net_position_minor"] / 100 for row in view.positions],
            marker_color=[row["color"] for row in view.positions],
            hovertemplate="%{x}<br>%{y:.2f} " + view.currency + "<extra></extra>",
        )
    )
    figure.update_layout(
        title="Participant net positions",
        yaxis_title=f"Net position ({view.currency} major units)",
        template="plotly_white",
        height=380,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
    )
    figure.add_hline(y=0, line_color="#142033", line_width=1)
    return figure


def status_figure(view: CycleView) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=list(view.status_counts.keys()),
            y=list(view.status_counts.values()),
            marker_color="#1f4b99",
        )
    )
    figure.update_layout(
        title="Transaction status",
        yaxis_title="Count",
        template="plotly_white",
        height=320,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
    )
    return figure


def probability_figure(view: CycleView) -> go.Figure:
    figure = go.Figure(
        go.Bar(
            x=[row["bank_id"] for row in view.probabilities],
            y=[
                0 if row["model_probability"] is None else row["model_probability"]
                for row in view.probabilities
            ],
            marker_color="#1f4b99",
            name="Model probability",
        )
    )
    figure.add_hline(
        y=view.display_threshold,
        line_dash="dash",
        line_color="#9b3a3a",
        annotation_text="Display threshold",
    )
    figure.update_layout(
        title="Predicted shortfall probability",
        yaxis_title="Probability",
        yaxis={"range": [0, 1]},
        template="plotly_white",
        height=360,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
    )
    return figure


def history_figure(view: CycleView) -> go.Figure:
    figure = go.Figure(
        go.Scatter(
            x=[row["cycle_id"] for row in view.history],
            y=[row["simulated_shortfalls"] for row in view.history],
            mode="lines+markers",
            line={"color": "#1f4b99"},
            name="Simulated shortfalls",
        )
    )
    figure.update_layout(
        title="Simulated shortfalls by cycle",
        yaxis_title="Banks with a shortfall",
        template="plotly_white",
        height=320,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
    )
    return figure


def _render(view: CycleView) -> None:
    st.warning(view.disclaimer)
    st.caption("Nothing on this page submits a payment instruction or moves funds.")
    if view.split_name is not None:
        st.caption(
            f"Cycle `{view.cycle_id}` is in the **{view.split_name}** portion of the model evaluation. "
            "Probabilities shown for training cycles are in-sample scores. The metrics table uses the test period."
        )
    if view.error:
        st.error(view.error)
    if view.reconciled:
        st.success(view.reconciliation_message)
    else:
        st.error(view.reconciliation_message or "Reconciliation status is unavailable.")

    st.subheader("Executive summary")
    first, second, third, fourth = st.columns(4)
    first.metric("Transactions", view.transaction_count)
    second.metric("Eligible bilateral amount", _money(view.eligible_amount_minor, view.currency))
    third.metric("Quarantined records", view.quarantined_count)
    fourth.metric("Late arrivals excluded", view.late_count)
    st.caption(
        f"Purchases included: {view.purchase_count}. Explicit reversals included: {view.reversal_count}. "
        f"Currency: {view.currency}."
    )

    st.subheader("Transaction status")
    if view.status_counts:
        st.plotly_chart(status_figure(view), use_container_width=True)
    else:
        st.info("This cycle has no transactions to chart.")

    st.subheader("Net positions")
    st.caption(
        "Negative bars are net debtors. Positive bars are net creditors. This is a simulated DNS result."
    )
    if view.positions:
        st.plotly_chart(position_figure(view), use_container_width=True)
        st.dataframe(
            pd.DataFrame(view.positions)[
                ["bank_id", "role", "net_position", "gross_debit", "gross_credit"]
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No participant positions were produced for this cycle.")

    st.subheader("Shortfall probabilities")
    st.caption(
        "The model probability is separate from the simulated outcome. "
        "The display threshold only highlights rows. It does not execute funding."
    )
    if view.probabilities:
        st.plotly_chart(probability_figure(view), use_container_width=True)
        st.dataframe(pd.DataFrame(view.probabilities), use_container_width=True, hide_index=True)
    else:
        st.info("No prediction snapshot is available for this cycle.")

    st.subheader("Funding recommendations")
    st.caption(
        "These buffers were not sent to a bank and do not change net settlement obligations."
    )
    if view.unresolved:
        st.error(
            "Unresolved liquidity risk remains for at least one bank. The cap was not pretended away."
        )
        st.dataframe(pd.DataFrame(view.unresolved), use_container_width=True, hide_index=True)
    if view.funding_rows:
        st.markdown("**Risk-informed recommendation**")
        st.dataframe(pd.DataFrame(view.funding_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No funding recommendation was produced for this cycle.")
    if view.strategy_rows:
        st.markdown("**Strategy comparison on the simulated outcome**")
        st.dataframe(pd.DataFrame(view.strategy_rows), use_container_width=True, hide_index=True)

    st.subheader("Model evaluation")
    if view.metrics_error:
        st.error(view.metrics_error)
    elif view.metrics_rows:
        st.caption(view.limitation or "")
        if view.selected_model:
            st.write(f"Model selected on validation F1: `{view.selected_model}`.")
        st.dataframe(pd.DataFrame(view.metrics_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Model metrics are not loaded. Generate an evaluation to fill this panel.")

    st.subheader("History across cycles")
    if view.history:
        st.plotly_chart(history_figure(view), use_container_width=True)
        st.dataframe(pd.DataFrame(view.history), use_container_width=True, hide_index=True)
    else:
        st.info("No historical cycles are loaded.")

    st.subheader("Transaction inspection")
    if not view.transactions:
        st.info("No transactions belong to this cycle.")
        return
    statuses = ["All", *sorted({row["status"] for row in view.transactions})]
    banks = [
        "All",
        *sorted(
            {row["issuer_bank"] for row in view.transactions}
            | {row["acquirer_bank"] for row in view.transactions}
        ),
    ]
    status_filter, bank_filter = st.columns(2)
    chosen_status = status_filter.selectbox("Status", statuses)
    chosen_bank = bank_filter.selectbox("Bank", banks)
    filtered = view.transactions
    if chosen_status != "All":
        filtered = [row for row in filtered if row["status"] == chosen_status]
    if chosen_bank != "All":
        filtered = [
            row
            for row in filtered
            if row["issuer_bank"] == chosen_bank or row["acquirer_bank"] == chosen_bank
        ]
    if not filtered:
        st.info("No transactions match this filter.")
    else:
        st.dataframe(pd.DataFrame(filtered), use_container_width=True, hide_index=True)


def main() -> None:
    """Launch the operations dashboard."""
    configure_logging("INFO")
    st.set_page_config(page_title="Smart Clearing & Settlement", layout="wide")
    st.markdown(
        """
        <style>
        h1, h2, h3 { letter-spacing: -0.02em; }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d5dde8;
            border-radius: 8px;
            padding: 12px 14px;
        }
        div[data-testid="stMetricLabel"], div[data-testid="stMetricValue"] {
            color: #142033;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Smart Clearing & Settlement Engine")
    st.caption(
        "Simulated card clearing, deferred net positions, shortfall scores, and funding buffers."
    )
    with st.sidebar:
        st.header("Simulation")
        seed = int(st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1))
        n_cycles = int(st.slider("Weekday cycles", min_value=12, max_value=24, value=18))
        display_threshold = float(
            st.slider(
                "Display risk threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05
            )
        )
        funding_limit = int(
            st.number_input(
                "Per-bank funding limit (minor units)", min_value=0, value=5_000_000, step=100_000
            )
        )
        cost_bps = int(
            st.number_input(
                "Funding cost (basis points)", min_value=0, max_value=10_000, value=25, step=5
            )
        )
        risk_limit = float(
            st.slider("Risk limit", min_value=0.0, max_value=1.0, value=0.20, step=0.05)
        )
        st.caption("Recommendations stay inside the engine. They are not payment instructions.")

    try:
        transactions, snapshots = cached_simulation(seed, n_cycles)
    except Exception as exc:
        st.error(f"The simulator failed: {exc}")
        return
    scores: pd.DataFrame | None = None
    metrics: dict[str, Any] | None = None
    metrics_error: str | None = None
    try:
        scores, metrics = cached_metrics(seed, n_cycles)
    except InsufficientDataError as exc:
        metrics_error = str(exc)
    except Exception as exc:
        metrics_error = f"Model evaluation failed: {exc}"

    cycle_ids = sorted(set(snapshots["cycle_id"]).union(set(transactions["cycle_id"])))
    if not cycle_ids:
        st.info("The simulation did not produce any settlement cycles.")
        return
    with st.sidebar:
        cycle_id = st.selectbox("Settlement cycle", cycle_ids, index=len(cycle_ids) - 1)
    view = build_cycle_view(
        transactions,
        snapshots,
        str(cycle_id),
        scores=scores,
        metrics=metrics,
        metrics_error=metrics_error,
        display_threshold=display_threshold,
        funding_limit_minor=funding_limit,
        liquidity_cost_bps=cost_bps,
        risk_limit=risk_limit,
    )
    _render(view)


if __name__ == "__main__":
    main()
