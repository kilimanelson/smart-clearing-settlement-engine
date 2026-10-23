"""Shared data contracts for clearing, simulation, the API, and the dashboard.

Monetary amounts are integer minor units. A value of ``100`` means 100 minor
units, which the display helper can render as 1.00 when the currency exponent
is 2. The contracts themselves never store major-unit floats.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ID_PATTERN = r"^[A-Za-z0-9_\-]+$"


class TransactionStatus(StrEnum):
    """Lifecycle state of a simulated card clearing record."""

    PENDING = "pending"
    CLEARED = "cleared"
    REVERSED = "reversed"


class CycleStatus(StrEnum):
    """Allowed simulation states for a settlement cycle."""

    OPEN = "open"
    CLEARED = "cleared"
    PREDICTED = "predicted"
    OPTIMIZED = "optimized"


class QuarantineReason(StrEnum):
    """Why a record was left out of a clearing cycle."""

    NON_POSITIVE_AMOUNT = "non_positive_amount"
    SAME_BANK = "same_bank"
    WRONG_CYCLE = "wrong_cycle"
    WRONG_CURRENCY = "wrong_currency"
    NOT_ELIGIBLE_STATUS = "not_eligible_status"
    REVERSAL_MISSING_REFERENCE = "reversal_missing_reference"
    CLEARED_HAS_REVERSAL_REFERENCE = "cleared_has_reversal_reference"


class FundingStrategy(StrEnum):
    """Named liquidity policies compared by the optimizer."""

    NO_BUFFER = "no_buffer"
    FIXED_PERCENTAGE = "fixed_percentage"
    RULES_BASED = "rules_based"
    RISK_INFORMED = "risk_informed"


def _reject_bool_and_float(value: object, field_name: str) -> object:
    if isinstance(value, bool) or isinstance(value, float):
        raise ValueError(f"{field_name} must be an integer number of minor units")
    return value


def ensure_utc(value: datetime) -> datetime:
    """Return ``value`` as an aware UTC datetime.

    Naive timestamps are rejected so cycle boundaries cannot silently shift.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class Transaction(_StrictModel):
    """One simulated card-payment clearing record.

    Invalid business combinations are representable here so the clearing engine
    can quarantine them. Request models used by the API are stricter.
    """

    transaction_id: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    cycle_id: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    issuer_bank: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    acquirer_bank: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    amount_minor: int
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    status: TransactionStatus
    event_timestamp: datetime
    clearing_timestamp: datetime | None = None
    reversal_of_transaction_id: str | None = Field(default=None, pattern=_ID_PATTERN)

    @field_validator("amount_minor", mode="before")
    @classmethod
    def _amount_is_minor_units(cls, value: object) -> object:
        return _reject_bool_and_float(value, "amount_minor")

    @field_validator("event_timestamp", "clearing_timestamp")
    @classmethod
    def _timestamps_are_aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc(value)


class TransactionCreate(Transaction):
    """Ingestion contract. Rejects records the simulation API should not store."""

    amount_minor: int = Field(gt=0)

    @model_validator(mode="after")
    def _status_links_are_consistent(self) -> TransactionCreate:
        if self.issuer_bank == self.acquirer_bank:
            raise ValueError("issuer_bank and acquirer_bank must differ")
        if self.status == TransactionStatus.REVERSED and not self.reversal_of_transaction_id:
            raise ValueError("reversed transactions require reversal_of_transaction_id")
        if self.status != TransactionStatus.REVERSED and self.reversal_of_transaction_id:
            raise ValueError("only reversed transactions may set reversal_of_transaction_id")
        return self


class TransactionBatch(_StrictModel):
    """A batch of transactions submitted together."""

    transactions: list[TransactionCreate] = Field(min_length=1, max_length=10_000)


class QuarantinedRecord(_StrictModel):
    """A record excluded from the obligations of one cycle."""

    transaction_id: str
    reason_code: QuarantineReason
    detail: str


class BilateralObligation(_StrictModel):
    """Net amount one participant owes another in a single currency."""

    debtor_bank: str
    creditor_bank: str
    amount_minor: int = Field(gt=0)
    currency: str


class NetPosition(_StrictModel):
    """Multilateral position of one participant.

    ``net_position_minor`` is negative when the participant owes money and
    positive when the participant is due to receive money.
    """

    bank_id: str
    net_position_minor: int
    gross_debit_minor: int = Field(ge=0)
    gross_credit_minor: int = Field(ge=0)
    currency: str


class ClearingReport(_StrictModel):
    """Settlement-cycle reconciliation report produced by the clearing engine."""

    cycle_id: str
    currency: str
    purchase_count: int = Field(ge=0)
    reversal_count: int = Field(ge=0)
    quarantined_count: int = Field(ge=0)
    bilateral_obligations: list[BilateralObligation]
    net_positions: list[NetPosition]
    net_position_sum_minor: int
    reconciled: bool
    quarantined: list[QuarantinedRecord]


class SnapshotInput(_StrictModel):
    """Pre-cutoff features for one bank, as supplied to prediction."""

    bank_id: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    prediction_timestamp: datetime
    cutoff_timestamp: datetime
    incoming_volume_so_far: int = Field(ge=0)
    outgoing_volume_so_far: int = Field(ge=0)
    transaction_count_so_far: int = Field(ge=0)
    average_transaction_amount_so_far: float = Field(ge=0)
    recent_volume_change: float
    historical_shortfall_rate: float = Field(ge=0, le=1)
    available_liquidity_at_prediction: int = Field(ge=0)
    predicted_or_forecast_net_debit: int = Field(ge=0)

    @field_validator(
        "incoming_volume_so_far",
        "outgoing_volume_so_far",
        "transaction_count_so_far",
        "available_liquidity_at_prediction",
        "predicted_or_forecast_net_debit",
        mode="before",
    )
    @classmethod
    def _integers(cls, value: object) -> object:
        return _reject_bool_and_float(value, "snapshot integer field")

    @field_validator("prediction_timestamp", "cutoff_timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return ensure_utc(value)

    @model_validator(mode="after")
    def _prediction_precedes_cutoff(self) -> SnapshotInput:
        if self.prediction_timestamp >= self.cutoff_timestamp:
            raise ValueError("prediction_timestamp must be earlier than cutoff_timestamp")
        return self


class PredictRequest(_StrictModel):
    """Snapshots used to score shortfall risk for an already cleared cycle."""

    snapshots: list[SnapshotInput] = Field(min_length=1, max_length=1_000)


class OptimizerConfig(_StrictModel):
    """Knobs for the documented funding policies.

    Costs are basis points of additional funding for the cycle, so 25 means
    0.25 percent. ``stress_fraction_bps`` of 5000 means a 50 percent add-on
    when the predicted probability is 1.
    """

    risk_limit: float = Field(default=0.2, ge=0, le=1)
    fixed_buffer_percent: int = Field(default=10, ge=0, le=100)
    rules_buffer_percent: int = Field(default=15, ge=0, le=100)
    stress_fraction_bps: int = Field(default=5_000, ge=0, le=10_000)
    portfolio_funding_cap_minor: int | None = Field(default=None, ge=0)

    @field_validator(
        "fixed_buffer_percent", "rules_buffer_percent", "stress_fraction_bps", mode="before"
    )
    @classmethod
    def _config_ints(cls, value: object) -> object:
        if isinstance(value, bool) or isinstance(value, float):
            raise ValueError("optimizer settings that count units must be integers")
        return value

    @field_validator("portfolio_funding_cap_minor", mode="before")
    @classmethod
    def _cap_int(cls, value: object) -> object:
        if value is None:
            return None
        return _reject_bool_and_float(value, "portfolio_funding_cap_minor")


class OptimizeRequest(OptimizerConfig):
    """Uniform funding constraints applied to every bank in a cycle."""

    liquidity_cost_bps: int = Field(default=25, ge=0, le=10_000)
    funding_limit_minor: int = Field(ge=0)

    @field_validator("liquidity_cost_bps", "funding_limit_minor", mode="before")
    @classmethod
    def _request_ints(cls, value: object) -> object:
        return _reject_bool_and_float(value, "optimize request integer field")


class LiquidityInput(_StrictModel):
    """One bank's inputs to the liquidity optimizer."""

    bank_id: str = Field(min_length=1, max_length=64, pattern=_ID_PATTERN)
    shortfall_probability: float = Field(ge=0, le=1)
    forecast_net_debit_minor: int = Field(ge=0)
    available_liquidity_minor: int = Field(ge=0)
    liquidity_cost_bps: int = Field(ge=0, le=10_000)
    funding_limit_minor: int = Field(ge=0)

    @field_validator(
        "forecast_net_debit_minor",
        "available_liquidity_minor",
        "liquidity_cost_bps",
        "funding_limit_minor",
        mode="before",
    )
    @classmethod
    def _minor_ints(cls, value: object) -> object:
        return _reject_bool_and_float(value, "liquidity input integer field")


class FundingRecommendation(_StrictModel):
    """Recommended additional funding. This is not a payment instruction."""

    bank_id: str
    strategy: FundingStrategy
    additional_funding_minor: int = Field(ge=0)
    requirement_minor: int = Field(ge=0)
    unresolved_risk: bool
    residual_gap_minor: int = Field(ge=0)
    incremental_cost_minor: int = Field(ge=0)


class OptimizationResult(_StrictModel):
    """Allocations for one strategy, after constraints have been checked."""

    strategy: FundingStrategy
    recommendations: list[FundingRecommendation]
    total_funding_minor: int = Field(ge=0)
    total_cost_minor: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    respects_portfolio_cap: bool
    objective: str


class HealthResponse(_StrictModel):
    status: str
    simulation_only: bool
    database: str
    disclaimer: str


class IngestResponse(_StrictModel):
    created: int
    existing: int
    transaction_ids: list[str]
    disclaimer: str


class CycleSummary(_StrictModel):
    cycle_id: str
    currency: str
    status: CycleStatus
    reconciled: bool | None
    transaction_count: int


class PredictionRecord(_StrictModel):
    bank_id: str
    model_version: str
    predicted_at: datetime
    probability: float
    predicted_label: int
    threshold: float


class PredictResponse(_StrictModel):
    cycle_id: str
    model_version: str
    predictions: list[PredictionRecord]
    disclaimer: str


class StrategyScoreCard(_StrictModel):
    strategy: FundingStrategy
    total_funding_minor: int
    total_cost_minor: int
    unresolved_count: int
    shortfall_count: int | None = None
    shortfall_rate: float | None = None
    mean_utilization: float | None = None


class OptimizeResponse(_StrictModel):
    cycle_id: str
    recommendations: list[FundingRecommendation]
    strategies: list[StrategyScoreCard]
    disclaimer: str
    note: str


class CycleReportResponse(_StrictModel):
    cycle_id: str
    currency: str
    status: CycleStatus
    reconciled: bool
    storage_matches_clearing: bool
    clearing: ClearingReport
    predictions: list[PredictionRecord]
    recommendations: list[FundingRecommendation]
    disclaimer: str


def format_minor_units(amount_minor: int, currency: str, *, exponent: int = 2) -> str:
    """Render integer minor units for display.

    The conversion is presentation only. Callers must keep calculating with
    ``amount_minor``.
    """
    if isinstance(amount_minor, bool) or not isinstance(amount_minor, int):
        raise TypeError("amount_minor must be an int")
    if exponent < 0:
        raise ValueError("exponent cannot be negative")
    sign = "-" if amount_minor < 0 else ""
    scale = 10**exponent
    absolute = abs(amount_minor)
    major = absolute // scale
    minor = absolute % scale
    if exponent == 0:
        return f"{sign}{major:,} {currency}"
    return f"{sign}{major:,}.{minor:0{exponent}d} {currency}"


def json_ready(value: Any) -> Any:
    """Recursively convert values that ``json`` can store in an audit payload."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value
