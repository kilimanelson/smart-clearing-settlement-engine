"""Liquidity buffer recommendations that do not change settlement obligations.

The cash objective reported to users is::

    incremental_cost_minor = additional_funding_minor * liquidity_cost_bps // 10000

The risk-informed policy asks a linear solver to minimize that charge plus a
penalty on uncovered requirement. The penalty is larger for a higher predicted
shortfall probability, so scarce portfolio funds go to the banks the model
rates as riskier. The penalty is not a cash cost and is not included in
``total_cost_minor``.

A predicted probability does not guarantee that the realized shortfall rate
will match it. When a bank cannot raise enough funds under the caps, the
result sets ``unresolved_risk`` and leaves the DNS position untouched.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Never

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from smart_clearing.schemas import (
    FundingRecommendation,
    FundingStrategy,
    LiquidityInput,
    OptimizationResult,
    OptimizerConfig,
)

logger = logging.getLogger(__name__)

OBJECTIVE = (
    "Minimize sum(liquidity_cost_bps * additional_funding_minor) plus a linear "
    "penalty on each bank's uncovered liquidity requirement. The penalty base "
    "is larger than the maximum allowed basis-point cost, and it increases with "
    "the predicted shortfall probability. Additional funding is bounded by the "
    "per-bank limit and by the gap to the requirement, and optionally by a "
    "portfolio cap."
)

RISK_MODEL_NOTE = (
    "The requirement is a transparent policy, not a value-at-risk guarantee. "
    "Below the configured probability limit, the requirement is the forecast "
    "net debit. At or above the limit, the rules-based and risk-informed "
    "policies add a buffer. A predicted probability alone does not mean the "
    "realized shortfall rate will equal that probability."
)

_PENALTY_BASE = 1_000_000


class OptimizerError(Exception):
    """The funding request is invalid or the solver did not return a usable point."""


def _assert_never(value: Never) -> None:
    raise OptimizerError(f"Unhandled funding strategy: {value}")


def incremental_cost_minor(funding_minor: int, cost_bps: int) -> int:
    """Cycle funding charge in minor units, using integer arithmetic."""
    if funding_minor < 0 or cost_bps < 0:
        raise OptimizerError("funding and cost cannot be negative")
    return funding_minor * cost_bps // 10_000


def liquidity_requirement_minor(
    strategy: FundingStrategy,
    item: LiquidityInput,
    config: OptimizerConfig,
) -> int:
    """Minor units of liquidity the strategy wants available at cutoff.

    Forecast net debit is the point requirement. Extra buffers apply only to
    the strategies that define one. ``no_buffer`` still *reports* the debit as
    the requirement and then recommends zero additional funding.
    """
    debit = item.forecast_net_debit_minor
    if strategy == FundingStrategy.NO_BUFFER:
        return debit
    if strategy == FundingStrategy.FIXED_PERCENTAGE:
        return debit + debit * config.fixed_buffer_percent // 100
    if strategy == FundingStrategy.RULES_BASED:
        if item.shortfall_probability >= config.risk_limit:
            return debit + debit * config.rules_buffer_percent // 100
        return debit
    if strategy == FundingStrategy.RISK_INFORMED:
        if item.shortfall_probability >= config.risk_limit:
            probability_bps = int(round(item.shortfall_probability * 10_000))
            addon = debit * probability_bps * config.stress_fraction_bps // (10_000 * 10_000)
            return debit + addon
        return debit
    _assert_never(strategy)


def _recommendation(
    strategy: FundingStrategy,
    item: LiquidityInput,
    funding_minor: int,
    requirement_minor: int,
) -> FundingRecommendation:
    residual = max(0, requirement_minor - item.available_liquidity_minor - funding_minor)
    return FundingRecommendation(
        bank_id=item.bank_id,
        strategy=strategy,
        additional_funding_minor=funding_minor,
        requirement_minor=requirement_minor,
        unresolved_risk=residual > 0,
        residual_gap_minor=residual,
        incremental_cost_minor=incremental_cost_minor(funding_minor, item.liquidity_cost_bps),
    )


def _rule_funding(
    strategy: FundingStrategy,
    items: Sequence[LiquidityInput],
    config: OptimizerConfig,
) -> list[FundingRecommendation]:
    recommendations: list[FundingRecommendation] = []
    for item in items:
        requirement = liquidity_requirement_minor(strategy, item, config)
        gap = max(0, requirement - item.available_liquidity_minor)
        funding = 0 if strategy == FundingStrategy.NO_BUFFER else min(item.funding_limit_minor, gap)
        recommendations.append(_recommendation(strategy, item, funding, requirement))
    return recommendations


def _repair_portfolio_cap(
    funding: list[int],
    items: Sequence[LiquidityInput],
    cap: int,
) -> list[int]:
    excess = sum(funding) - cap
    if excess <= 0:
        return funding
    order = sorted(
        range(len(items)),
        key=lambda index: (
            items[index].shortfall_probability,
            -items[index].liquidity_cost_bps,
            items[index].bank_id,
        ),
    )
    repaired = list(funding)
    for index in order:
        reduction = min(repaired[index], excess)
        repaired[index] -= reduction
        excess -= reduction
        if excess == 0:
            break
    if excess > 0:
        raise OptimizerError("Portfolio cap repair could not remove the excess funding")
    return repaired


def _solve_risk_informed(items: Sequence[LiquidityInput], config: OptimizerConfig) -> list[int]:
    gaps: list[int] = []
    requirements: list[int] = []
    for item in items:
        requirement = liquidity_requirement_minor(FundingStrategy.RISK_INFORMED, item, config)
        requirements.append(requirement)
        gaps.append(max(0, requirement - item.available_liquidity_minor))
    width = len(items)
    objective = np.zeros(width * 2, dtype=float)
    upper = np.zeros(width * 2, dtype=float)
    for index, item in enumerate(items):
        objective[index] = float(item.liquidity_cost_bps)
        objective[width + index] = float(
            _PENALTY_BASE + int(round(item.shortfall_probability * _PENALTY_BASE))
        )
        upper[index] = float(min(item.funding_limit_minor, gaps[index]))
        upper[width + index] = float(gaps[index])
    rows: list[np.ndarray] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for index, gap in enumerate(gaps):
        row = np.zeros(width * 2, dtype=float)
        row[index] = 1.0
        row[width + index] = 1.0
        rows.append(row)
        lower_bounds.append(float(gap))
        upper_bounds.append(np.inf)
    if config.portfolio_funding_cap_minor is not None:
        cap_row = np.zeros(width * 2, dtype=float)
        cap_row[:width] = 1.0
        rows.append(cap_row)
        lower_bounds.append(0.0)
        upper_bounds.append(float(config.portfolio_funding_cap_minor))
    solved = milp(
        objective,
        integrality=np.ones(width * 2),
        bounds=Bounds(np.zeros(width * 2), upper),
        constraints=LinearConstraint(np.vstack(rows), lower_bounds, upper_bounds),
        options={"time_limit": 30},
    )
    if not solved.success or solved.x is None:
        message = getattr(solved, "message", "unknown solver failure")
        raise OptimizerError(f"Liquidity optimization failed: {message}")
    funding = [int(round(float(solved.x[index]))) for index in range(width)]
    clamped: list[int] = []
    for amount, item, gap in zip(funding, items, gaps, strict=True):
        clamped.append(min(min(item.funding_limit_minor, gap), max(0, amount)))
    if config.portfolio_funding_cap_minor is not None:
        clamped = _repair_portfolio_cap(clamped, items, config.portfolio_funding_cap_minor)
    return clamped


def assert_funding_constraints(
    result: OptimizationResult,
    items: Sequence[LiquidityInput],
    config: OptimizerConfig,
) -> None:
    """Raise if a returned allocation breaks a documented constraint."""
    by_bank = {item.bank_id: item for item in items}
    if len(by_bank) != len(items):
        raise OptimizerError("Duplicate bank ids in optimizer inputs")
    if {row.bank_id for row in result.recommendations} != set(by_bank):
        raise OptimizerError("Recommendations do not match the input banks")
    total_funding = 0
    total_cost = 0
    unresolved = 0
    for recommendation in result.recommendations:
        item = by_bank[recommendation.bank_id]
        if recommendation.strategy != result.strategy:
            raise OptimizerError("Recommendation strategy does not match the result")
        expected_requirement = liquidity_requirement_minor(result.strategy, item, config)
        if recommendation.requirement_minor != expected_requirement:
            raise OptimizerError(f"{item.bank_id} requirement does not match the policy")
        if recommendation.additional_funding_minor < 0:
            raise OptimizerError("Additional funding cannot be negative")
        if recommendation.additional_funding_minor > item.funding_limit_minor:
            raise OptimizerError(f"{item.bank_id} exceeds its funding limit")
        if (
            result.strategy == FundingStrategy.NO_BUFFER
            and recommendation.additional_funding_minor != 0
        ):
            raise OptimizerError("The no-buffer strategy must not add funding")
        covered = item.available_liquidity_minor + recommendation.additional_funding_minor
        expected_residual = max(0, recommendation.requirement_minor - covered)
        if recommendation.residual_gap_minor != expected_residual:
            raise OptimizerError(f"{item.bank_id} residual gap is inconsistent")
        if recommendation.unresolved_risk != (expected_residual > 0):
            raise OptimizerError(f"{item.bank_id} unresolved flag is inconsistent")
        expected_cost = incremental_cost_minor(
            recommendation.additional_funding_minor,
            item.liquidity_cost_bps,
        )
        if recommendation.incremental_cost_minor != expected_cost:
            raise OptimizerError(f"{item.bank_id} cost is inconsistent")
        total_funding += recommendation.additional_funding_minor
        total_cost += recommendation.incremental_cost_minor
        unresolved += int(recommendation.unresolved_risk)
    if total_funding != result.total_funding_minor or total_cost != result.total_cost_minor:
        raise OptimizerError("Totals do not reconcile to the bank recommendations")
    if unresolved != result.unresolved_count:
        raise OptimizerError("Unresolved count does not reconcile")
    cap = config.portfolio_funding_cap_minor
    if result.strategy == FundingStrategy.RISK_INFORMED and cap is not None and total_funding > cap:
        raise OptimizerError("Risk-informed funding exceeds the portfolio cap")


def optimize_liquidity(
    inputs: Sequence[LiquidityInput],
    config: OptimizerConfig | None = None,
    strategy: FundingStrategy = FundingStrategy.RISK_INFORMED,
) -> OptimizationResult:
    """Recommend additional funding for ``strategy``.

    The input sequence is not modified. Net settlement positions are not an
    input and cannot be rewritten here.
    """
    resolved = config or OptimizerConfig()
    if len({item.bank_id for item in inputs}) != len(inputs):
        raise OptimizerError("Duplicate bank ids in optimizer inputs")
    ordered = sorted(inputs, key=lambda item: item.bank_id)
    if strategy == FundingStrategy.RISK_INFORMED and ordered:
        funding = _solve_risk_informed(ordered, resolved)
        recommendations = [
            _recommendation(
                strategy,
                item,
                amount,
                liquidity_requirement_minor(strategy, item, resolved),
            )
            for item, amount in zip(ordered, funding, strict=True)
        ]
    elif strategy == FundingStrategy.RISK_INFORMED:
        recommendations = []
    elif strategy in (
        FundingStrategy.NO_BUFFER,
        FundingStrategy.FIXED_PERCENTAGE,
        FundingStrategy.RULES_BASED,
    ):
        recommendations = _rule_funding(strategy, ordered, resolved)
    else:
        _assert_never(strategy)
    result = OptimizationResult(
        strategy=strategy,
        recommendations=recommendations,
        total_funding_minor=sum(item.additional_funding_minor for item in recommendations),
        total_cost_minor=sum(item.incremental_cost_minor for item in recommendations),
        unresolved_count=sum(int(item.unresolved_risk) for item in recommendations),
        respects_portfolio_cap=strategy == FundingStrategy.RISK_INFORMED
        or resolved.portfolio_funding_cap_minor is None,
        objective=OBJECTIVE,
    )
    assert_funding_constraints(result, ordered, resolved)
    logger.info(
        "Optimized %s banks=%d funding=%d unresolved=%d",
        strategy.value,
        len(recommendations),
        result.total_funding_minor,
        result.unresolved_count,
    )
    return result


def compare_funding_strategies(
    inputs: Sequence[LiquidityInput],
    config: OptimizerConfig | None = None,
) -> list[OptimizationResult]:
    """Run the four documented policies on the same inputs."""
    resolved = config or OptimizerConfig()
    return [optimize_liquidity(inputs, resolved, strategy) for strategy in FundingStrategy]


@dataclass(frozen=True)
class ActualLiquidityOutcome:
    """Ex-post simulated debit and cutoff liquidity. Not an optimizer input."""

    bank_id: str
    actual_net_debit_minor: int
    liquidity_at_cutoff_minor: int


@dataclass(frozen=True)
class StrategyEvaluation:
    """How a frozen recommendation would have performed on simulated outcomes."""

    strategy: FundingStrategy
    total_funding_minor: int
    total_cost_minor: int
    unresolved_count: int
    shortfall_count: int
    shortfall_rate: float
    mean_utilization: float | None


def evaluate_strategy(
    result: OptimizationResult,
    actuals: Sequence[ActualLiquidityOutcome],
) -> StrategyEvaluation:
    """Score recommendations against outcomes the optimizer was not shown.

    Added funding is treated as cash that arrives before cutoff. It is added
    to cutoff liquidity and is not subtracted from the DNS net debit.
    """
    by_bank = {item.bank_id: item for item in actuals}
    if set(by_bank) != {row.bank_id for row in result.recommendations}:
        raise OptimizerError("Actual outcomes must cover exactly the recommended banks")
    shortfalls = 0
    utilizations: list[float] = []
    for recommendation in result.recommendations:
        actual = by_bank[recommendation.bank_id]
        if actual.actual_net_debit_minor < 0 or actual.liquidity_at_cutoff_minor < 0:
            raise OptimizerError("Actual debit and cutoff liquidity cannot be negative")
        capacity = actual.liquidity_at_cutoff_minor + recommendation.additional_funding_minor
        if actual.actual_net_debit_minor > capacity:
            shortfalls += 1
        if capacity > 0:
            utilizations.append(min(1.0, actual.actual_net_debit_minor / capacity))
    count = len(result.recommendations)
    return StrategyEvaluation(
        strategy=result.strategy,
        total_funding_minor=result.total_funding_minor,
        total_cost_minor=result.total_cost_minor,
        unresolved_count=result.unresolved_count,
        shortfall_count=shortfalls,
        shortfall_rate=(shortfalls / count) if count else 0.0,
        mean_utilization=(sum(utilizations) / len(utilizations)) if utilizations else None,
    )
