"""One simulated run from transactions through clearing, scoring, and funding."""

from smart_clearing.app import build_cycle_view, comparison_to_dict
from smart_clearing.clearing import clear_cycle
from smart_clearing.features import MODEL_FEATURES, feature_matrix
from smart_clearing.model import score_snapshots, train_and_evaluate
from smart_clearing.optimizer import optimize_liquidity
from smart_clearing.schemas import LiquidityInput
from smart_clearing.simulator import demo_config, frame_to_transactions, simulate


def test_simulated_cycle_clears_scores_and_recommends_without_changing_positions() -> None:
    result = simulate(demo_config())
    cycle_id = str(result.snapshots["cycle_id"].iloc[-1])
    currency = result.config.currency
    cycle_rows = result.transactions.loc[result.transactions["cycle_id"] == cycle_id]
    banks = sorted(
        result.snapshots.loc[result.snapshots["cycle_id"] == cycle_id, "bank_id"].unique()
    )
    report = clear_cycle(
        frame_to_transactions(cycle_rows),
        cycle_id=cycle_id,
        currency=currency,
        participants=banks,
    )
    assert report.reconciled is True
    assert report.net_position_sum_minor == 0
    matrix = feature_matrix(result.snapshots)
    assert list(matrix.columns) == list(MODEL_FEATURES)
    trained = train_and_evaluate(result.snapshots, seed=result.config.seed)
    assert len(trained.report.runs) == 3
    assert max(trained.report.train_cycles) < min(trained.report.test_cycles)
    sample = result.snapshots.loc[result.snapshots["cycle_id"] == cycle_id].iloc[0]
    before = report.model_dump()
    recommendation = optimize_liquidity(
        [
            LiquidityInput(
                bank_id=str(sample["bank_id"]),
                shortfall_probability=0.4,
                forecast_net_debit_minor=int(sample["predicted_or_forecast_net_debit"]),
                available_liquidity_minor=int(sample["available_liquidity_at_prediction"]),
                liquidity_cost_bps=25,
                funding_limit_minor=5_000_000,
            )
        ]
    )
    assert report.model_dump() == before
    assert recommendation.total_funding_minor >= 0
    view = build_cycle_view(
        result.transactions,
        result.snapshots,
        cycle_id,
        scores=score_snapshots(trained.model, result.snapshots),
        metrics=comparison_to_dict(trained.report),
    )
    assert view.reconciled is True
    assert view.positions
    assert "not production-ready" in view.disclaimer
