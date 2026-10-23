"""Funding policies, caps, and the promise not to rewrite settlement positions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from smart_clearing.optimizer import (
    ActualLiquidityOutcome,
    OptimizerError,
    compare_funding_strategies,
    evaluate_strategy,
    incremental_cost_minor,
    optimize_liquidity,
)
from smart_clearing.schemas import FundingStrategy, LiquidityInput, NetPosition, OptimizerConfig


def _bank(bank_id: str, **overrides: object) -> LiquidityInput:
    payload: dict[str, object] = {
        "bank_id": bank_id,
        "shortfall_probability": 0.1,
        "forecast_net_debit_minor": 1_000,
        "available_liquidity_minor": 1_000,
        "liquidity_cost_bps": 25,
        "funding_limit_minor": 10_000,
    }
    payload.update(overrides)
    return LiquidityInput(**payload)  # type: ignore[arg-type]


def test_cost_uses_integer_basis_points() -> None:
    assert incremental_cost_minor(10_000, 25) == 25


def test_zero_debit_needs_no_funding() -> None:
    result = optimize_liquidity(
        [_bank("A", forecast_net_debit_minor=0, available_liquidity_minor=500)],
        OptimizerConfig(),
        FundingStrategy.RISK_INFORMED,
    )
    assert result.total_funding_minor == 0
    assert result.unresolved_count == 0
    assert result.recommendations[0].unresolved_risk is False


def test_fixed_percentage_and_no_buffer_are_transparent() -> None:
    item = _bank(
        "A",
        forecast_net_debit_minor=1_000,
        available_liquidity_minor=900,
        funding_limit_minor=5_000,
    )
    config = OptimizerConfig(fixed_buffer_percent=10)
    no_buffer = optimize_liquidity([item], config, FundingStrategy.NO_BUFFER)
    fixed = optimize_liquidity([item], config, FundingStrategy.FIXED_PERCENTAGE)
    assert no_buffer.recommendations[0].additional_funding_minor == 0
    assert no_buffer.recommendations[0].unresolved_risk is True
    assert no_buffer.recommendations[0].residual_gap_minor == 100
    assert fixed.recommendations[0].requirement_minor == 1_100
    assert fixed.recommendations[0].additional_funding_minor == 200
    assert fixed.recommendations[0].unresolved_risk is False


def test_insufficient_limit_is_unresolved_rather_than_silently_covered() -> None:
    result = optimize_liquidity(
        [
            _bank(
                "A",
                forecast_net_debit_minor=500,
                available_liquidity_minor=0,
                funding_limit_minor=0,
            )
        ],
        OptimizerConfig(risk_limit=1),
        FundingStrategy.RISK_INFORMED,
    )
    recommendation = result.recommendations[0]
    assert recommendation.additional_funding_minor == 0
    assert recommendation.unresolved_risk is True
    assert recommendation.residual_gap_minor == 500


def test_zero_cost_does_not_fund_beyond_the_requirement() -> None:
    result = optimize_liquidity(
        [
            _bank(
                "A",
                forecast_net_debit_minor=1_000,
                available_liquidity_minor=0,
                liquidity_cost_bps=0,
                funding_limit_minor=5_000,
                shortfall_probability=0.0,
            )
        ],
        OptimizerConfig(risk_limit=1),
        FundingStrategy.RISK_INFORMED,
    )
    assert result.recommendations[0].additional_funding_minor == 1_000


def test_portfolio_cap_funds_the_higher_risk_bank_and_flags_the_other() -> None:
    inputs = [
        _bank(
            "A",
            shortfall_probability=0.9,
            forecast_net_debit_minor=100,
            available_liquidity_minor=0,
            liquidity_cost_bps=50,
        ),
        _bank(
            "B",
            shortfall_probability=0.1,
            forecast_net_debit_minor=100,
            available_liquidity_minor=0,
            liquidity_cost_bps=1,
        ),
    ]
    result = optimize_liquidity(
        inputs, OptimizerConfig(risk_limit=1, portfolio_funding_cap_minor=100)
    )
    by_bank = {item.bank_id: item for item in result.recommendations}
    assert by_bank["A"].additional_funding_minor == 100
    assert by_bank["B"].additional_funding_minor == 0
    assert by_bank["B"].unresolved_risk is True
    assert result.total_funding_minor == 100


def test_invalid_constraints_are_rejected() -> None:
    with pytest.raises(ValidationError):
        _bank("A", shortfall_probability=1.5)
    with pytest.raises(ValidationError):
        _bank("A", funding_limit_minor=-1)
    with pytest.raises(ValidationError):
        _bank("A", available_liquidity_minor=-5)


def test_solver_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "smart_clearing.optimizer.milp",
        lambda *_args, **_kwargs: SimpleNamespace(success=False, message="infeasible", x=None),
    )
    with pytest.raises(OptimizerError, match="failed"):
        optimize_liquidity([_bank("A", forecast_net_debit_minor=100, available_liquidity_minor=0)])


def test_recommendations_are_reproducible_and_do_not_change_positions() -> None:
    positions = [
        NetPosition(
            bank_id="A",
            net_position_minor=-1000,
            gross_debit_minor=1000,
            gross_credit_minor=0,
            currency="USD",
        ),
        NetPosition(
            bank_id="B",
            net_position_minor=1000,
            gross_debit_minor=0,
            gross_credit_minor=1000,
            currency="USD",
        ),
    ]
    before = [item.model_dump() for item in positions]
    inputs = [
        _bank(
            "A",
            forecast_net_debit_minor=1_000,
            available_liquidity_minor=400,
            shortfall_probability=0.8,
        )
    ]
    config = OptimizerConfig(risk_limit=0.2)
    first = compare_funding_strategies(inputs, config)
    second = compare_funding_strategies(inputs, config)
    assert [item.model_dump() for item in first] == [item.model_dump() for item in second]
    assert [item.model_dump() for item in positions] == before
    assert {item.strategy for item in first} == set(FundingStrategy)


def test_ex_post_score_uses_cutoff_liquidity_and_leaves_the_debit_unchanged() -> None:
    result = optimize_liquidity(
        [
            _bank(
                "A",
                forecast_net_debit_minor=100,
                available_liquidity_minor=0,
                funding_limit_minor=50,
                shortfall_probability=0.0,
            )
        ],
        OptimizerConfig(risk_limit=1),
        FundingStrategy.NO_BUFFER,
    )
    outcome = ActualLiquidityOutcome(
        bank_id="A", actual_net_debit_minor=100, liquidity_at_cutoff_minor=40
    )
    scored = evaluate_strategy(result, [outcome])
    assert scored.shortfall_count == 1
    assert outcome.actual_net_debit_minor == 100
    funded = optimize_liquidity(
        [
            _bank(
                "A",
                forecast_net_debit_minor=100,
                available_liquidity_minor=0,
                funding_limit_minor=70,
                shortfall_probability=0.0,
            )
        ],
        OptimizerConfig(risk_limit=1),
        FundingStrategy.FIXED_PERCENTAGE,
    )
    # Fixed percentage adds 10 percent, and liquidity at prediction is zero, so funding is 110,
    # which is above the 70 limit and therefore unresolved. Score the no-buffer case above
    # and a fully funded rules result separately.
    assert funded.recommendations[0].additional_funding_minor == 70
    covered = optimize_liquidity(
        [
            _bank(
                "A",
                forecast_net_debit_minor=100,
                available_liquidity_minor=0,
                funding_limit_minor=100,
                shortfall_probability=0.0,
            )
        ],
        OptimizerConfig(risk_limit=1, fixed_buffer_percent=0),
        FundingStrategy.FIXED_PERCENTAGE,
    )
    assert evaluate_strategy(covered, [outcome]).shortfall_count == 0
