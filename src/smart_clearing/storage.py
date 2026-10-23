"""PostgreSQL persistence for the simulation.

The application database is PostgreSQL. SQLite is refused unless
``SCSE_ALLOW_SQLITE`` is set, and that switch exists only so the automated
tests can run without a database service. Nothing in this module submits a
payment or connects to a card network.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    event,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from smart_clearing.clearing import clear_cycle
from smart_clearing.config import SIMULATION_DISCLAIMER, Settings
from smart_clearing.model import SavedShortfallModel
from smart_clearing.optimizer import RISK_MODEL_NOTE, compare_funding_strategies
from smart_clearing.schemas import (
    ClearingReport,
    CycleReportResponse,
    CycleStatus,
    CycleSummary,
    FundingRecommendation,
    FundingStrategy,
    LiquidityInput,
    NetPosition,
    OptimizeRequest,
    OptimizeResponse,
    PredictionRecord,
    PredictResponse,
    SnapshotInput,
    StrategyScoreCard,
    Transaction,
    TransactionCreate,
    TransactionStatus,
    json_ready,
)

logger = logging.getLogger(__name__)


class StorageError(Exception):
    """Base class for persistence failures."""


class StorageConfigError(StorageError):
    """The database URL is missing or points at a database the app will not use."""


class NotFoundError(StorageError):
    """The requested cycle does not exist."""


class PayloadConflictError(StorageError):
    """An idempotent write saw a different payload for an existing id."""


class InvalidStateError(StorageError):
    """The cycle is not in a state that allows the requested transition."""


class Base(DeclarativeBase):
    """Declarative base for simulation tables."""


class BankRow(Base):
    __tablename__ = "banks"

    bank_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SettlementCycleRow(Base):
    __tablename__ = "settlement_cycles"

    cycle_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    prediction_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cutoff_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reconciled: Mapped[bool | None] = mapped_column(Boolean)
    prediction_input_hash: Mapped[str | None] = mapped_column(String(64))
    optimization_input_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class TransactionRow(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    cycle_id: Mapped[str] = mapped_column(
        ForeignKey("settlement_cycles.cycle_id"), nullable=False, index=True
    )
    issuer_bank: Mapped[str] = mapped_column(
        ForeignKey("banks.bank_id"), nullable=False, index=True
    )
    acquirer_bank: Mapped[str] = mapped_column(
        ForeignKey("banks.bank_id"), nullable=False, index=True
    )
    amount_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    clearing_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    reversal_of_transaction_id: Mapped[str | None] = mapped_column(String(64))
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class NetPositionRow(Base):
    __tablename__ = "net_positions"
    __table_args__ = (UniqueConstraint("cycle_id", "bank_id", name="uq_position_cycle_bank"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(
        ForeignKey("settlement_cycles.cycle_id"), nullable=False, index=True
    )
    bank_id: Mapped[str] = mapped_column(ForeignKey("banks.bank_id"), nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False)
    net_position_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    gross_debit_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    gross_credit_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)


class LiquiditySnapshotRow(Base):
    __tablename__ = "liquidity_snapshots"
    __table_args__ = (UniqueConstraint("cycle_id", "bank_id", name="uq_liquidity_cycle_bank"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(
        ForeignKey("settlement_cycles.cycle_id"), nullable=False, index=True
    )
    bank_id: Mapped[str] = mapped_column(ForeignKey("banks.bank_id"), nullable=False)
    prediction_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cutoff_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_liquidity_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    forecast_net_debit_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    incoming_volume_so_far: Mapped[int] = mapped_column(BigInteger, nullable=False)
    outgoing_volume_so_far: Mapped[int] = mapped_column(BigInteger, nullable=False)
    transaction_count_so_far: Mapped[int] = mapped_column(Integer, nullable=False)
    average_transaction_amount_so_far: Mapped[float] = mapped_column(nullable=False)
    recent_volume_change: Mapped[float] = mapped_column(nullable=False)
    historical_shortfall_rate: Mapped[float] = mapped_column(nullable=False)


class RiskPredictionRow(Base):
    __tablename__ = "risk_predictions"
    __table_args__ = (
        UniqueConstraint("cycle_id", "bank_id", "model_version", name="uq_prediction_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(
        ForeignKey("settlement_cycles.cycle_id"), nullable=False, index=True
    )
    bank_id: Mapped[str] = mapped_column(ForeignKey("banks.bank_id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    probability: Mapped[float] = mapped_column(nullable=False)
    predicted_label: Mapped[int] = mapped_column(Integer, nullable=False)
    threshold: Mapped[float] = mapped_column(nullable=False)


class FundingRecommendationRow(Base):
    __tablename__ = "funding_recommendations"
    __table_args__ = (
        UniqueConstraint("cycle_id", "bank_id", "strategy", name="uq_funding_strategy"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(
        ForeignKey("settlement_cycles.cycle_id"), nullable=False, index=True
    )
    bank_id: Mapped[str] = mapped_column(ForeignKey("banks.bank_id"), nullable=False)
    strategy: Mapped[str] = mapped_column(String(64), nullable=False)
    additional_funding_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    requirement_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    unresolved_risk: Mapped[bool] = mapped_column(Boolean, nullable=False)
    residual_gap_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    incremental_cost_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SimulatedSettlementResultRow(Base):
    __tablename__ = "simulated_settlement_results"
    __table_args__ = (UniqueConstraint("cycle_id", "bank_id", name="uq_simulated_result"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[str] = mapped_column(
        ForeignKey("settlement_cycles.cycle_id"), nullable=False, index=True
    )
    bank_id: Mapped[str] = mapped_column(ForeignKey("banks.bank_id"), nullable=False)
    actual_net_debit_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    net_position_minor: Mapped[int] = mapped_column(BigInteger, nullable=False)
    actual_shortfall: Mapped[int | None] = mapped_column(Integer)
    liquidity_at_cutoff_minor: Mapped[int | None] = mapped_column(BigInteger)


class AuditEventRow(Base):
    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    cycle_id: Mapped[str | None] = mapped_column(String(64), index=True)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


def create_db_engine(settings: Settings) -> Engine:
    """Open the configured database, or explain how to start PostgreSQL."""
    url = settings.database_url.strip()
    if not url:
        raise StorageConfigError(
            "SCSE_DATABASE_URL is not set. Copy .env.example to .env and start PostgreSQL "
            "with `docker compose up postgres`."
        )
    if url.startswith("sqlite"):
        if not settings.allow_sqlite:
            raise StorageConfigError(
                "SQLite is not the application database. Use the PostgreSQL service in "
                "docker-compose.yml. SCSE_ALLOW_SQLITE=1 is only for automated tests."
            )
        engine = create_engine(url, future=True, connect_args={"check_same_thread": False})
        _enable_sqlite_foreign_keys(engine)
        return engine
    if not url.startswith("postgresql"):
        raise StorageConfigError("SCSE_DATABASE_URL must be a PostgreSQL SQLAlchemy URL.")
    return create_engine(url, future=True, pool_pre_ping=True)


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def init_database(engine: Engine) -> None:
    """Create any missing simulation tables."""
    Base.metadata.create_all(engine)


def check_database(engine: Engine) -> bool:
    """Return whether a trivial query succeeds. This does not validate financial data."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        logger.warning("Database health check failed", exc_info=True)
        return False
    return True


def fingerprint_payload(payload: dict[str, Any]) -> str:
    """Stable hash of a JSON-compatible payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _stored_datetime(value: datetime) -> datetime:
    """Restore UTC on values that SQLite returned without a timezone."""
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _to_transaction(row: TransactionRow) -> Transaction:
    return Transaction(
        transaction_id=row.transaction_id,
        cycle_id=row.cycle_id,
        issuer_bank=row.issuer_bank,
        acquirer_bank=row.acquirer_bank,
        amount_minor=int(row.amount_minor),
        currency=row.currency,
        status=TransactionStatus(row.status),
        event_timestamp=_stored_datetime(row.event_timestamp),
        clearing_timestamp=None
        if row.clearing_timestamp is None
        else _stored_datetime(row.clearing_timestamp),
        reversal_of_transaction_id=row.reversal_of_transaction_id,
    )


class Repository:
    """Transactional read and write methods. The caller commits."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def ingest(self, transactions: list[TransactionCreate]) -> tuple[int, int, list[str]]:
        """Insert transactions. Replaying an identical payload is a no-op."""
        identifiers = [item.transaction_id for item in transactions]
        if len(identifiers) != len(set(identifiers)):
            raise PayloadConflictError("The batch contains duplicate transaction ids")
        created = 0
        existing = 0
        for item in transactions:
            self._ensure_bank(item.issuer_bank)
            self._ensure_bank(item.acquirer_bank)
            cycle = self._ensure_cycle(item.cycle_id, item.currency)
            digest = fingerprint_payload(item.model_dump(mode="json"))
            current = self.session.get(TransactionRow, item.transaction_id)
            if current is not None:
                if current.payload_hash != digest:
                    raise PayloadConflictError(
                        f"Transaction {item.transaction_id} already exists with a different payload"
                    )
                existing += 1
                continue
            if cycle.status != CycleStatus.OPEN.value:
                raise InvalidStateError(
                    f"Cycle {item.cycle_id} is {cycle.status} and cannot accept new transactions"
                )
            self.session.add(
                TransactionRow(
                    transaction_id=item.transaction_id,
                    cycle_id=item.cycle_id,
                    issuer_bank=item.issuer_bank,
                    acquirer_bank=item.acquirer_bank,
                    amount_minor=item.amount_minor,
                    currency=item.currency,
                    status=item.status.value,
                    event_timestamp=item.event_timestamp,
                    clearing_timestamp=item.clearing_timestamp,
                    reversal_of_transaction_id=item.reversal_of_transaction_id,
                    payload_hash=digest,
                    created_at=_utc_now(),
                )
            )
            created += 1
        self._audit(
            "transactions.ingested",
            transactions[0].cycle_id
            if len({item.cycle_id for item in transactions}) == 1
            else None,
            {"created": created, "existing": existing, "transaction_ids": identifiers},
        )
        logger.info("Ingested transactions created=%d existing=%d", created, existing)
        return created, existing, identifiers

    def list_cycles(self) -> list[CycleSummary]:
        cycles = self.session.scalars(
            select(SettlementCycleRow).order_by(SettlementCycleRow.cycle_id)
        ).all()
        summaries: list[CycleSummary] = []
        for cycle in cycles:
            count = self.session.scalars(
                select(TransactionRow).where(TransactionRow.cycle_id == cycle.cycle_id)
            ).all()
            summaries.append(
                CycleSummary(
                    cycle_id=cycle.cycle_id,
                    currency=cycle.currency,
                    status=CycleStatus(cycle.status),
                    reconciled=cycle.reconciled,
                    transaction_count=len(count),
                )
            )
        return summaries

    def clear(self, cycle_id: str) -> ClearingReport:
        """Clear an open cycle and store positions. A later call returns the same obligations."""
        cycle = self._require_cycle(cycle_id)
        records = [_to_transaction(row) for row in self._transactions(cycle_id)]
        participants = sorted(
            {record.issuer_bank for record in records}
            | {record.acquirer_bank for record in records}
        )
        report = clear_cycle(
            records, cycle_id=cycle_id, currency=cycle.currency, participants=participants
        )
        if cycle.status == CycleStatus.OPEN.value:
            for position in report.net_positions:
                self._ensure_bank(position.bank_id)
                self.session.add(
                    NetPositionRow(
                        cycle_id=cycle_id,
                        bank_id=position.bank_id,
                        currency=position.currency,
                        net_position_minor=position.net_position_minor,
                        gross_debit_minor=position.gross_debit_minor,
                        gross_credit_minor=position.gross_credit_minor,
                    )
                )
                debit = -position.net_position_minor if position.net_position_minor < 0 else 0
                self.session.add(
                    SimulatedSettlementResultRow(
                        cycle_id=cycle_id,
                        bank_id=position.bank_id,
                        actual_net_debit_minor=debit,
                        net_position_minor=position.net_position_minor,
                        actual_shortfall=None,
                        liquidity_at_cutoff_minor=None,
                    )
                )
            cycle.status = CycleStatus.CLEARED.value
            cycle.reconciled = True
            self._audit(
                "cycle.cleared",
                cycle_id,
                {
                    "purchase_count": report.purchase_count,
                    "reversal_count": report.reversal_count,
                    "quarantined_count": report.quarantined_count,
                    "net_position_sum_minor": report.net_position_sum_minor,
                },
            )
            logger.info("Stored clearing results for cycle %s", cycle_id)
        return report

    def positions(self, cycle_id: str) -> list[NetPosition]:
        cycle = self._require_cycle(cycle_id)
        if cycle.status == CycleStatus.OPEN.value:
            raise InvalidStateError(f"Cycle {cycle_id} has not been cleared")
        rows = self.session.scalars(
            select(NetPositionRow)
            .where(NetPositionRow.cycle_id == cycle_id)
            .order_by(NetPositionRow.bank_id)
        ).all()
        return [
            NetPosition(
                bank_id=row.bank_id,
                net_position_minor=int(row.net_position_minor),
                gross_debit_minor=int(row.gross_debit_minor),
                gross_credit_minor=int(row.gross_credit_minor),
                currency=row.currency,
            )
            for row in rows
        ]

    def predict(
        self,
        cycle_id: str,
        snapshots: list[SnapshotInput],
        model: SavedShortfallModel,
    ) -> PredictResponse:
        """Score a cleared cycle. The first snapshot payload is kept; a different one conflicts."""
        cycle = self._require_cycle(cycle_id)
        if cycle.status == CycleStatus.OPEN.value:
            raise InvalidStateError("Risk prediction requires a cleared cycle")
        self._require_common_clock(snapshots)
        ordered_snapshots = sorted(snapshots, key=lambda item: item.bank_id)
        digest = fingerprint_payload([item.model_dump(mode="json") for item in ordered_snapshots])
        if cycle.status in {CycleStatus.PREDICTED.value, CycleStatus.OPTIMIZED.value}:
            if cycle.prediction_input_hash != digest:
                raise InvalidStateError(
                    "Predictions are already stored for a different snapshot payload"
                )
            return self._prediction_response(cycle)
        frame = pd.DataFrame([item.model_dump() for item in ordered_snapshots])
        probabilities = model.predict_proba_positive(frame)
        labels = (probabilities >= model.threshold).astype(int)
        predicted_at = _utc_now()
        for index, snapshot in enumerate(ordered_snapshots):
            self._ensure_bank(snapshot.bank_id)
            self.session.add(
                LiquiditySnapshotRow(
                    cycle_id=cycle_id,
                    bank_id=snapshot.bank_id,
                    prediction_timestamp=snapshot.prediction_timestamp,
                    cutoff_timestamp=snapshot.cutoff_timestamp,
                    available_liquidity_minor=snapshot.available_liquidity_at_prediction,
                    forecast_net_debit_minor=snapshot.predicted_or_forecast_net_debit,
                    incoming_volume_so_far=snapshot.incoming_volume_so_far,
                    outgoing_volume_so_far=snapshot.outgoing_volume_so_far,
                    transaction_count_so_far=snapshot.transaction_count_so_far,
                    average_transaction_amount_so_far=snapshot.average_transaction_amount_so_far,
                    recent_volume_change=snapshot.recent_volume_change,
                    historical_shortfall_rate=snapshot.historical_shortfall_rate,
                )
            )
            self.session.add(
                RiskPredictionRow(
                    cycle_id=cycle_id,
                    bank_id=snapshot.bank_id,
                    model_version=model.model_version,
                    predicted_at=predicted_at,
                    probability=float(probabilities[index]),
                    predicted_label=int(labels[index]),
                    threshold=model.threshold,
                )
            )
        cycle.status = CycleStatus.PREDICTED.value
        cycle.prediction_input_hash = digest
        cycle.prediction_timestamp = snapshots[0].prediction_timestamp
        cycle.cutoff_timestamp = snapshots[0].cutoff_timestamp
        self._audit(
            "cycle.predict_risk",
            cycle_id,
            {
                "model_version": model.model_version,
                "banks": [item.bank_id for item in ordered_snapshots],
            },
        )
        self.session.flush()
        logger.info("Stored risk predictions for cycle %s model %s", cycle_id, model.model_version)
        return self._prediction_response(cycle)

    def optimize(self, cycle_id: str, request: OptimizeRequest) -> OptimizeResponse:
        """Store funding recommendations without changing net positions."""
        cycle = self._require_cycle(cycle_id)
        if cycle.status not in {CycleStatus.PREDICTED.value, CycleStatus.OPTIMIZED.value}:
            raise InvalidStateError("Liquidity optimization requires stored risk predictions")
        digest = fingerprint_payload(request.model_dump(mode="json"))
        if cycle.status == CycleStatus.OPTIMIZED.value:
            if cycle.optimization_input_hash != digest:
                raise InvalidStateError(
                    "Funding recommendations are already stored for different constraints"
                )
            return self._optimize_response(cycle)
        predictions = self._predictions(cycle_id)
        liquidity = {
            row.bank_id: row
            for row in self.session.scalars(
                select(LiquiditySnapshotRow).where(LiquiditySnapshotRow.cycle_id == cycle_id)
            ).all()
        }
        inputs: list[LiquidityInput] = []
        for prediction in predictions:
            snapshot = liquidity.get(prediction.bank_id)
            if snapshot is None:
                raise InvalidStateError(f"No liquidity snapshot is stored for {prediction.bank_id}")
            inputs.append(
                LiquidityInput(
                    bank_id=prediction.bank_id,
                    shortfall_probability=float(prediction.probability),
                    forecast_net_debit_minor=int(snapshot.forecast_net_debit_minor),
                    available_liquidity_minor=int(snapshot.available_liquidity_minor),
                    liquidity_cost_bps=request.liquidity_cost_bps,
                    funding_limit_minor=request.funding_limit_minor,
                )
            )
        results = compare_funding_strategies(inputs, request)
        created_at = _utc_now()
        for result in results:
            for recommendation in result.recommendations:
                self.session.add(
                    FundingRecommendationRow(
                        cycle_id=cycle_id,
                        bank_id=recommendation.bank_id,
                        strategy=recommendation.strategy.value,
                        additional_funding_minor=recommendation.additional_funding_minor,
                        requirement_minor=recommendation.requirement_minor,
                        unresolved_risk=recommendation.unresolved_risk,
                        residual_gap_minor=recommendation.residual_gap_minor,
                        incremental_cost_minor=recommendation.incremental_cost_minor,
                        created_at=created_at,
                    )
                )
        cycle.status = CycleStatus.OPTIMIZED.value
        cycle.optimization_input_hash = digest
        self.session.flush()
        self._audit(
            "cycle.optimize_liquidity",
            cycle_id,
            {
                "strategies": [result.strategy.value for result in results],
                "note": "Recommendations were stored. No settlement position was changed.",
            },
        )
        logger.info("Stored funding recommendations for cycle %s", cycle_id)
        return self._optimize_response(cycle)

    def report(self, cycle_id: str) -> CycleReportResponse:
        """Recompute clearing and compare it with the stored positions."""
        cycle = self._require_cycle(cycle_id)
        if cycle.status == CycleStatus.OPEN.value:
            raise InvalidStateError(f"Cycle {cycle_id} has not been cleared")
        records = [_to_transaction(row) for row in self._transactions(cycle_id)]
        participants = sorted(
            {record.issuer_bank for record in records}
            | {record.acquirer_bank for record in records}
        )
        clearing = clear_cycle(
            records, cycle_id=cycle_id, currency=cycle.currency, participants=participants
        )
        stored = {
            position.bank_id: position.net_position_minor for position in self.positions(cycle_id)
        }
        recomputed = {
            position.bank_id: position.net_position_minor for position in clearing.net_positions
        }
        predictions = (
            [self._prediction_record(row) for row in self._predictions(cycle_id)]
            if cycle.status in {CycleStatus.PREDICTED.value, CycleStatus.OPTIMIZED.value}
            else []
        )
        recommendations = (
            self._recommendation_models(cycle_id)
            if cycle.status == CycleStatus.OPTIMIZED.value
            else []
        )
        return CycleReportResponse(
            cycle_id=cycle_id,
            currency=cycle.currency,
            status=CycleStatus(cycle.status),
            reconciled=bool(cycle.reconciled) and clearing.reconciled,
            storage_matches_clearing=stored == recomputed,
            clearing=clearing,
            predictions=predictions,
            recommendations=recommendations,
            disclaimer=SIMULATION_DISCLAIMER,
        )

    def _ensure_bank(self, bank_id: str) -> None:
        if self.session.get(BankRow, bank_id) is None:
            self.session.add(BankRow(bank_id=bank_id, created_at=_utc_now()))
            self.session.flush()

    def _ensure_cycle(self, cycle_id: str, currency: str) -> SettlementCycleRow:
        cycle = self.session.get(SettlementCycleRow, cycle_id)
        if cycle is None:
            cycle = SettlementCycleRow(
                cycle_id=cycle_id,
                currency=currency,
                status=CycleStatus.OPEN.value,
                reconciled=None,
                created_at=_utc_now(),
            )
            self.session.add(cycle)
            self.session.flush()
            return cycle
        if cycle.currency != currency:
            raise PayloadConflictError(
                f"Cycle {cycle_id} is {cycle.currency} and cannot accept {currency} transactions"
            )
        return cycle

    def _require_cycle(self, cycle_id: str) -> SettlementCycleRow:
        cycle = self.session.get(SettlementCycleRow, cycle_id)
        if cycle is None:
            raise NotFoundError(f"Cycle {cycle_id} was not found")
        return cycle

    def _transactions(self, cycle_id: str) -> list[TransactionRow]:
        return list(
            self.session.scalars(
                select(TransactionRow)
                .where(TransactionRow.cycle_id == cycle_id)
                .order_by(TransactionRow.transaction_id)
            ).all()
        )

    def _predictions(self, cycle_id: str) -> list[RiskPredictionRow]:
        return list(
            self.session.scalars(
                select(RiskPredictionRow)
                .where(RiskPredictionRow.cycle_id == cycle_id)
                .order_by(RiskPredictionRow.bank_id)
            ).all()
        )

    def _audit(self, action: str, cycle_id: str | None, detail: dict[str, Any]) -> None:
        self.session.add(
            AuditEventRow(
                created_at=_utc_now(),
                action=action,
                cycle_id=cycle_id,
                detail=json_ready(detail),
            )
        )

    def _require_common_clock(self, snapshots: list[SnapshotInput]) -> None:
        prediction_times = {item.prediction_timestamp for item in snapshots}
        cutoff_times = {item.cutoff_timestamp for item in snapshots}
        if len(prediction_times) != 1 or len(cutoff_times) != 1:
            raise InvalidStateError(
                "All snapshots in a cycle must share prediction and cutoff timestamps"
            )
        banks = [item.bank_id for item in snapshots]
        if len(banks) != len(set(banks)):
            raise PayloadConflictError("Snapshot batch contains duplicate bank ids")

    def _prediction_record(self, row: RiskPredictionRow) -> PredictionRecord:
        return PredictionRecord(
            bank_id=row.bank_id,
            model_version=row.model_version,
            predicted_at=_stored_datetime(row.predicted_at),
            probability=float(row.probability),
            predicted_label=int(row.predicted_label),
            threshold=float(row.threshold),
        )

    def _prediction_response(self, cycle: SettlementCycleRow) -> PredictResponse:
        records = [self._prediction_record(row) for row in self._predictions(cycle.cycle_id)]
        version = records[0].model_version if records else "unknown"
        return PredictResponse(
            cycle_id=cycle.cycle_id,
            model_version=version,
            predictions=records,
            disclaimer=SIMULATION_DISCLAIMER,
        )

    def _recommendation_models(self, cycle_id: str) -> list[FundingRecommendation]:
        rows = self.session.scalars(
            select(FundingRecommendationRow)
            .where(FundingRecommendationRow.cycle_id == cycle_id)
            .order_by(FundingRecommendationRow.strategy, FundingRecommendationRow.bank_id)
        ).all()
        return [
            FundingRecommendation(
                bank_id=row.bank_id,
                strategy=FundingStrategy(row.strategy),
                additional_funding_minor=int(row.additional_funding_minor),
                requirement_minor=int(row.requirement_minor),
                unresolved_risk=bool(row.unresolved_risk),
                residual_gap_minor=int(row.residual_gap_minor),
                incremental_cost_minor=int(row.incremental_cost_minor),
            )
            for row in rows
        ]

    def _optimize_response(self, cycle: SettlementCycleRow) -> OptimizeResponse:
        recommendations = self._recommendation_models(cycle.cycle_id)
        grouped: dict[FundingStrategy, list[FundingRecommendation]] = {
            strategy: [] for strategy in FundingStrategy
        }
        for item in recommendations:
            grouped[item.strategy].append(item)
        cards = [
            StrategyScoreCard(
                strategy=strategy,
                total_funding_minor=sum(
                    item.additional_funding_minor for item in grouped[strategy]
                ),
                total_cost_minor=sum(item.incremental_cost_minor for item in grouped[strategy]),
                unresolved_count=sum(int(item.unresolved_risk) for item in grouped[strategy]),
                shortfall_count=None,
                shortfall_rate=None,
                mean_utilization=None,
            )
            for strategy in FundingStrategy
        ]
        return OptimizeResponse(
            cycle_id=cycle.cycle_id,
            recommendations=recommendations,
            strategies=cards,
            disclaimer=SIMULATION_DISCLAIMER,
            note=RISK_MODEL_NOTE
            + " Ex-post shortfall counts are omitted because cutoff liquidity was not supplied.",
        )
