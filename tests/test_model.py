"""Chronological evaluation of the baseline, logistic regression, and XGBoost."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from smart_clearing.features import MODEL_FEATURES, chronological_cycle_split, feature_matrix
from smart_clearing.model import (
    EstimatorKind,
    InsufficientDataError,
    ModelInputError,
    load_model,
    render_results_markdown,
    save_model,
    score_snapshots,
    train_and_evaluate,
)
from smart_clearing.simulator import demo_config, simulate


@pytest.fixture(scope="module")
def training():
    """One demo training run shared by the model tests in this module."""
    result = simulate(demo_config())
    return result.snapshots, train_and_evaluate(result.snapshots, seed=42)


def test_splits_are_temporal_and_all_three_models_are_scored(training) -> None:
    snapshots, trained = training
    report = trained.report
    assert [run.name for run in report.runs] == [kind.value for kind in EstimatorKind]
    assert max(report.train_cycles) < min(report.validation_cycles)
    assert max(report.validation_cycles) < min(report.test_cycles)
    assert set(report.train_cycles).isdisjoint(report.test_cycles)
    train, _validation, test = chronological_cycle_split(snapshots)
    pipeline = trained.estimators[EstimatorKind.LOGISTIC.value]
    scaler_mean = pipeline.named_steps["scaler"].mean_
    np.testing.assert_allclose(scaler_mean, feature_matrix(train).mean().to_numpy())
    full_mean = feature_matrix(pd.concat([train, test], ignore_index=True)).mean().to_numpy()
    assert not np.allclose(scaler_mean, full_mean)
    for run in report.runs:
        assert run.test.support_positive > 0
        assert run.test.false_positive >= 0
        assert run.test.false_negative >= 0
        assert "synthetic" in report.limitation.lower() or "not a live" in report.limitation.lower()


def test_training_is_reproducible(training) -> None:
    snapshots, first = training
    second = train_and_evaluate(snapshots, seed=42)
    assert first.report.selected_model == second.report.selected_model
    for left, right in zip(first.report.runs, second.report.runs, strict=True):
        assert left.test.f1 == right.test.f1
        assert left.threshold == right.threshold
        assert left.test.brier == pytest.approx(right.test.brier)


def test_saved_model_reloads_and_validates_its_schema(tmp_path, training) -> None:
    snapshots, trained = training
    path = tmp_path / "shortfall_model.joblib"
    save_model(trained.model, path)
    loaded = load_model(path)
    original = score_snapshots(trained.model, snapshots)
    restored = score_snapshots(loaded, snapshots)
    pd.testing.assert_frame_equal(original, restored)
    assert list(loaded.feature_names) == list(MODEL_FEATURES)
    broken = snapshots.drop(columns=["available_liquidity_at_prediction"])
    with pytest.raises(ModelInputError):
        loaded.predict_proba_positive(broken)


def test_results_markdown_uses_the_measured_scores(training) -> None:
    _snapshots, trained = training
    text = render_results_markdown(trained.report)
    first = trained.report.runs[0]
    assert first.name in text
    assert f"{first.test.f1:.3f}" in text
    assert "xgboost" in text
    assert "logistic_regression" in text
    assert "not validated banking performance" in text.lower() or "synthetic" in text.lower()


def test_a_one_class_sample_is_reported_instead_of_invented(training) -> None:
    snapshots, _trained = training
    only_negative = snapshots.loc[snapshots["actual_shortfall_at_cutoff"] == 0]
    if only_negative["cycle_id"].nunique() < 3:
        pytest.skip("Demo sample has fewer than three all-negative cycles.")
    with pytest.raises(InsufficientDataError):
        train_and_evaluate(only_negative, seed=42)
