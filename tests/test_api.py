"""API validation, idempotency, and the simulated cycle state machine."""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from smart_clearing.api import create_app
from smart_clearing.config import get_settings, reset_settings_cache
from smart_clearing.model import save_model, unfitted_rules_model
from smart_clearing.storage import AuditEventRow, StorageConfigError, create_db_engine

_EVENT = "2024-01-02T10:00:00Z"
_PREDICTION = "2024-01-02T14:00:00Z"
_CUTOFF = "2024-01-02T17:00:00Z"


def _transaction(transaction_id: str = "T1", **overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "transaction_id": transaction_id,
        "cycle_id": "C1",
        "issuer_bank": "A",
        "acquirer_bank": "B",
        "amount_minor": 1000,
        "currency": "USD",
        "status": "cleared",
        "event_timestamp": _EVENT,
    }
    payload.update(overrides)
    return payload


def _snapshot(bank_id: str, outgoing: int) -> dict[str, object]:
    return {
        "bank_id": bank_id,
        "prediction_timestamp": _PREDICTION,
        "cutoff_timestamp": _CUTOFF,
        "incoming_volume_so_far": 0,
        "outgoing_volume_so_far": outgoing,
        "transaction_count_so_far": 1,
        "average_transaction_amount_so_far": float(outgoing),
        "recent_volume_change": 0.0,
        "historical_shortfall_rate": 0.0,
        "available_liquidity_at_prediction": 100,
        "predicted_or_forecast_net_debit": outgoing,
    }


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch):
    database = tmp_path / "scse.sqlite"
    monkeypatch.setenv("SCSE_DATABASE_URL", f"sqlite:///{database}")
    monkeypatch.setenv("SCSE_ALLOW_SQLITE", "1")
    monkeypatch.setenv("SCSE_MODEL_PATH", str(tmp_path / "missing.joblib"))
    monkeypatch.setenv("SCSE_LOG_LEVEL", "WARNING")
    reset_settings_cache()
    application = create_app(get_settings())
    with TestClient(application) as test_client:
        yield test_client
    reset_settings_cache()


def test_sqlite_is_opt_in_and_a_missing_url_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCSE_DATABASE_URL", "sqlite:///unused.db")
    monkeypatch.setenv("SCSE_ALLOW_SQLITE", "0")
    reset_settings_cache()
    with pytest.raises(StorageConfigError):
        create_db_engine(get_settings())
    monkeypatch.setenv("SCSE_DATABASE_URL", "")
    reset_settings_cache()
    with pytest.raises(StorageConfigError):
        create_db_engine(get_settings())
    reset_settings_cache()


def test_health_and_input_validation(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["simulation_only"] is True
    assert body["database"] == "ok"
    assert "not production-ready" in body["disclaimer"]
    assert (
        client.post(
            "/transactions", json={"transactions": [_transaction(amount_minor=0)]}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/transactions",
            json={"transactions": [_transaction(issuer_bank="A", acquirer_bank="A")]},
        ).status_code
        == 422
    )


def test_cycle_state_machine_is_idempotent_and_does_not_rewrite_positions(
    client: TestClient,
) -> None:
    created = client.post(
        "/transactions",
        json={
            "transactions": [
                _transaction("T1", amount_minor=1000),
                _transaction("T2", issuer_bank="B", acquirer_bank="C", amount_minor=400),
            ]
        },
    )
    assert created.status_code == 200
    assert created.json()["created"] == 2
    replay = client.post(
        "/transactions",
        json={
            "transactions": [
                _transaction("T1", amount_minor=1000),
                _transaction("T2", issuer_bank="B", acquirer_bank="C", amount_minor=400),
            ]
        },
    )
    assert replay.status_code == 200
    assert replay.json()["existing"] == 2
    conflict = client.post(
        "/transactions", json={"transactions": [_transaction("T1", amount_minor=5)]}
    )
    assert conflict.status_code == 409
    duplicate_batch = client.post(
        "/transactions",
        json={"transactions": [_transaction("T9"), _transaction("T9", amount_minor=20)]},
    )
    assert duplicate_batch.status_code == 409
    assert client.get("/cycles/C1/positions").status_code == 409
    assert (
        client.post(
            "/cycles/C1/predict-risk", json={"snapshots": [_snapshot("A", 1000)]}
        ).status_code
        == 409
    )

    cleared = client.post("/cycles/C1/clear")
    assert cleared.status_code == 200
    positions = {
        row["bank_id"]: row["net_position_minor"] for row in cleared.json()["net_positions"]
    }
    assert sum(positions.values()) == 0
    assert positions["A"] == -1000
    assert positions["B"] == 600
    assert positions["C"] == 400
    again = client.post("/cycles/C1/clear")
    assert again.status_code == 200
    assert again.json()["net_position_sum_minor"] == 0
    assert (
        client.post(
            "/transactions", json={"transactions": [_transaction("T3", amount_minor=50)]}
        ).status_code
        == 409
    )

    stored = client.get("/cycles/C1/positions")
    assert stored.status_code == 200
    snapshots = {"snapshots": [_snapshot("A", 1000), _snapshot("B", 400), _snapshot("C", 0)]}
    predicted = client.post("/cycles/C1/predict-risk", json=snapshots)
    assert predicted.status_code == 200
    assert predicted.json()["model_version"] == "rules-baseline-fixed-scale"
    assert all(0.0 <= row["probability"] <= 1.0 for row in predicted.json()["predictions"])
    replayed = client.post("/cycles/C1/predict-risk", json=snapshots)
    assert replayed.status_code == 200
    assert replayed.json()["predictions"] == predicted.json()["predictions"]
    changed = dict(snapshots)
    changed["snapshots"] = [_snapshot("A", 5), _snapshot("B", 400), _snapshot("C", 0)]
    assert client.post("/cycles/C1/predict-risk", json=changed).status_code == 409

    optimize_body = {"liquidity_cost_bps": 25, "funding_limit_minor": 100_000, "risk_limit": 0.2}
    optimized = client.post("/cycles/C1/optimize-liquidity", json=optimize_body)
    assert optimized.status_code == 200
    assert (
        "not a value-at-risk guarantee" in optimized.json()["note"].lower()
        or "does not" in optimized.json()["note"].lower()
    )
    assert optimized.json()["disclaimer"]
    strategies = {row["strategy"] for row in optimized.json()["strategies"]}
    assert strategies == {"no_buffer", "fixed_percentage", "rules_based", "risk_informed"}
    assert client.get("/cycles/C1/positions").json() == stored.json()
    report = client.get("/cycles/C1/report")
    assert report.status_code == 200
    payload = report.json()
    assert payload["reconciled"] is True
    assert payload["storage_matches_clearing"] is True
    assert "not production-ready" in payload["disclaimer"]
    assert payload["clearing"]["net_position_sum_minor"] == 0

    session = client.app.state.session_factory()
    try:
        actions = set(session.scalars(select(AuditEventRow.action)).all())
    finally:
        session.close()
    assert {
        "transactions.ingested",
        "cycle.cleared",
        "cycle.predict_risk",
        "cycle.optimize_liquidity",
    } <= actions


def test_saved_artifact_is_the_predictor_version(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "model.joblib"
    save_model(unfitted_rules_model(), path)
    monkeypatch.setenv("SCSE_DATABASE_URL", f"sqlite:///{tmp_path / 'artifact.sqlite'}")
    monkeypatch.setenv("SCSE_ALLOW_SQLITE", "1")
    monkeypatch.setenv("SCSE_MODEL_PATH", str(path))
    monkeypatch.setenv("SCSE_LOG_LEVEL", "WARNING")
    reset_settings_cache()
    with TestClient(create_app(get_settings())) as test_client:
        assert (
            test_client.post("/transactions", json={"transactions": [_transaction()]}).status_code
            == 200
        )
        assert test_client.post("/cycles/C1/clear").status_code == 200
        response = test_client.post(
            "/cycles/C1/predict-risk", json={"snapshots": [_snapshot("A", 1000)]}
        )
    reset_settings_cache()
    assert response.status_code == 200
    assert response.json()["model_version"] == "rules-baseline-fixed-scale"


def test_postgres_round_trip_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    url = os.environ.get("SCSE_POSTGRES_TEST_URL")
    if not url:
        pytest.skip(
            "PostgreSQL is not configured. Start docker compose and set SCSE_POSTGRES_TEST_URL."
        )
    cycle_id = f"CY-{uuid.uuid4().hex[:12]}"
    monkeypatch.setenv("SCSE_DATABASE_URL", url)
    monkeypatch.setenv("SCSE_ALLOW_SQLITE", "0")
    monkeypatch.setenv("SCSE_LOG_LEVEL", "WARNING")
    reset_settings_cache()
    with TestClient(create_app(get_settings())) as test_client:
        ingested = test_client.post(
            "/transactions",
            json={
                "transactions": [
                    _transaction("PG1", cycle_id=cycle_id),
                    _transaction(
                        "PG2",
                        cycle_id=cycle_id,
                        issuer_bank="B",
                        acquirer_bank="A",
                        amount_minor=250,
                    ),
                ]
            },
        )
        assert ingested.status_code == 200
        cleared = test_client.post(f"/cycles/{cycle_id}/clear")
        assert cleared.status_code == 200
        assert cleared.json()["net_position_sum_minor"] == 0
    reset_settings_cache()
