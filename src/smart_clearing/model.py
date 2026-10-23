"""Shortfall classifiers fit only on information known at prediction time.

Three approaches are compared: a liquidity-gap baseline, L2-logistic
regression, and an XGBoost classifier. Thresholds and the selected model come
from the validation cycles. Test cycles are scored once for the comparison
table and are not used to choose a threshold.

Synthetic scores are not a measurement of real-bank risk. A predicted
probability is not a guarantee of the future shortfall rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Never

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from smart_clearing.features import (
    MODEL_FEATURES,
    assert_no_leakage,
    chronological_cycle_split,
    feature_matrix,
)

logger = logging.getLogger(__name__)

LIMITATION = (
    "Metrics describe a synthetic simulator, not a live card network. "
    "They are not validated banking performance and do not justify using "
    "this prototype for real settlement, capital, or liquidity decisions. "
    "Feature gain and logistic coefficients are associations in the simulated "
    "sample, not evidence of causality. SHAP was not added."
)


class ModelError(Exception):
    """The training or prediction request cannot be completed."""


class InsufficientDataError(ModelError):
    """The simulated sample does not support an honest classification metric."""


class ModelInputError(ModelError):
    """A prediction frame does not match the saved feature contract."""


class EstimatorKind(StrEnum):
    """The three approaches compared in every evaluation."""

    RULES = "rules_baseline"
    LOGISTIC = "logistic_regression"
    XGBOOST = "xgboost"


_COMPLEXITY = {
    EstimatorKind.RULES: 0,
    EstimatorKind.LOGISTIC: 1,
    EstimatorKind.XGBOOST: 2,
}


def _assert_never(value: Never) -> None:
    raise ModelError(f"Unhandled estimator kind: {value}")


class LiquidityGapBaseline:
    """Probability rises with forecast debit minus liquidity known now.

    The scale is the median absolute gap on the training cycles. A fixed scale
    can be supplied for an API fallback that has not seen a training sample.
    """

    def __init__(self, scale: float | None = None) -> None:
        self.scale_ = scale

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> LiquidityGapBaseline:
        del y
        gap = _gap(frame)
        scale = float(np.median(np.abs(gap)))
        self.scale_ = scale if scale > 0 else 1.0
        return self

    def predict_proba_positive(self, frame: pd.DataFrame) -> np.ndarray:
        if self.scale_ is None or self.scale_ <= 0:
            raise ModelError("Rules baseline must be fit before prediction")
        return 1.0 / (1.0 + np.exp(-_gap(frame) / self.scale_))


def _gap(frame: pd.DataFrame) -> np.ndarray:
    return frame["predicted_or_forecast_net_debit"].to_numpy(dtype=float) - frame[
        "available_liquidity_at_prediction"
    ].to_numpy(dtype=float)


@dataclass(frozen=True)
class ClassificationMetrics:
    """One evaluation of one model on one split."""

    split: str
    average_precision: float | None
    recall: float
    precision: float
    f1: float
    brier: float
    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int
    threshold: float
    support_positive: int
    support_negative: int


@dataclass(frozen=True)
class ModelRun:
    """Validation and one-shot test metrics for one approach."""

    name: str
    threshold: float
    calibrated: bool
    validation: ClassificationMetrics
    test: ClassificationMetrics
    feature_importances: dict[str, float] | None
    explanation: str
    stress_slices: dict[str, ClassificationMetrics]


@dataclass(frozen=True)
class ComparisonReport:
    """Side-by-side results. ``selected_model`` was chosen on validation F1."""

    runs: tuple[ModelRun, ...]
    selected_model: str
    seed: int
    train_cycles: tuple[str, ...]
    validation_cycles: tuple[str, ...]
    test_cycles: tuple[str, ...]
    train_positive_rate: float
    test_positive_rate: float
    limitation: str


class SavedShortfallModel:
    """A persisted predictor with an explicit feature contract and threshold."""

    def __init__(
        self,
        *,
        model_version: str,
        model_name: str,
        feature_names: list[str],
        threshold: float,
        seed: int,
        estimator_kind: EstimatorKind,
        estimator: object,
        isotonic: IsotonicRegression | None,
    ) -> None:
        self.model_version = model_version
        self.model_name = model_name
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        self.seed = int(seed)
        self.estimator_kind = estimator_kind
        self.estimator = estimator
        self.isotonic = isotonic

    @property
    def calibrated(self) -> bool:
        return self.isotonic is not None

    def predict_proba_positive(self, frame: pd.DataFrame) -> np.ndarray:
        """Return P(shortfall) for each row, after the saved calibration if any."""
        features = validate_feature_frame(frame, self.feature_names)
        raw = _positive_probability(self.estimator_kind, self.estimator, features)
        if self.isotonic is not None:
            raw = np.asarray(self.isotonic.predict(raw), dtype=float)
        return np.clip(raw, 0.0, 1.0)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Apply the validation-selected threshold. 1 means a predicted shortfall."""
        return (self.predict_proba_positive(frame) >= self.threshold).astype(int)


@dataclass
class TrainingResult:
    """A comparison report plus the model chosen on the validation split."""

    report: ComparisonReport
    model: SavedShortfallModel
    estimators: dict[str, object]


def validate_feature_frame(
    frame: pd.DataFrame, feature_names: list[str] | tuple[str, ...]
) -> pd.DataFrame:
    """Keep the saved columns, reject missing values, and reject leakage names."""
    if not isinstance(frame, pd.DataFrame):
        raise ModelInputError("Predictions require a pandas DataFrame")
    assert_no_leakage(feature_names)
    missing = [name for name in feature_names if name not in frame.columns]
    if missing:
        raise ModelInputError(f"Missing feature columns: {missing}")
    selected = frame.loc[:, list(feature_names)].copy()
    for column in selected.columns:
        if not pd.api.types.is_numeric_dtype(selected[column]):
            raise ModelInputError(f"{column} must be numeric")
    if selected.isna().to_numpy().any():
        raise ModelInputError("Features contain missing values")
    return selected


def unfitted_rules_model() -> SavedShortfallModel:
    """Gap rule used when no trained artifact is available.

    The scale is 100,000 minor units. That is a documented fallback, not a
    model that has been calibrated on a sample.
    """
    return SavedShortfallModel(
        model_version="rules-baseline-fixed-scale",
        model_name=EstimatorKind.RULES.value,
        feature_names=list(MODEL_FEATURES),
        threshold=0.5,
        seed=0,
        estimator_kind=EstimatorKind.RULES,
        estimator=LiquidityGapBaseline(scale=100_000.0),
        isotonic=None,
    )


def _require_both_classes(y: np.ndarray, split_name: str, *, minimum_positives: int) -> None:
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)
    if positives < minimum_positives or negatives < 1:
        raise InsufficientDataError(
            f"{split_name} split has {positives} shortfalls and {negatives} non-shortfalls. "
            "Increase the number of cycles or the funding-shock frequency. "
            "Metrics were not manufactured to fill the gap."
        )


def _positive_probability(
    kind: EstimatorKind, estimator: object, frame: pd.DataFrame
) -> np.ndarray:
    if kind == EstimatorKind.RULES:
        if not isinstance(estimator, LiquidityGapBaseline):
            raise ModelError("Rules artifact does not contain a liquidity-gap baseline")
        return np.asarray(estimator.predict_proba_positive(frame), dtype=float)
    if kind == EstimatorKind.LOGISTIC:
        if not isinstance(estimator, Pipeline):
            raise ModelError("Logistic artifact does not contain a sklearn pipeline")
        return np.asarray(estimator.predict_proba(frame)[:, 1], dtype=float)
    if kind == EstimatorKind.XGBOOST:
        if not isinstance(estimator, XGBClassifier):
            raise ModelError("XGBoost artifact does not contain an XGBClassifier")
        return np.asarray(estimator.predict_proba(frame)[:, 1], dtype=float)
    _assert_never(kind)


def _classification_metrics(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    *,
    split: str,
) -> ClassificationMetrics:
    predicted = (proba >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    average_precision: float | None
    if len(set(y_true.tolist())) < 2:
        average_precision = None
    else:
        average_precision = float(average_precision_score(y_true, proba))
    return ClassificationMetrics(
        split=split,
        average_precision=average_precision,
        recall=float(recall_score(y_true, predicted, zero_division=0)),
        precision=float(precision_score(y_true, predicted, zero_division=0)),
        f1=float(f1_score(y_true, predicted, zero_division=0)),
        brier=float(brier_score_loss(y_true, proba)),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
        threshold=threshold,
        support_positive=int(np.sum(y_true)),
        support_negative=int(len(y_true) - np.sum(y_true)),
    )


def select_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Pick the threshold with the highest validation F1.

    Ties keep the lower threshold, which favors catching more simulated
    shortfalls. The test set is not an input to this function.
    """
    best_threshold = 0.5
    best_f1 = -1.0
    for step in range(1, 20):
        threshold = step / 20
        score = float(f1_score(y_true, (proba >= threshold).astype(int), zero_division=0))
        if score > best_f1 + 1e-12:
            best_f1 = score
            best_threshold = threshold
    return best_threshold


def _maybe_calibrate(
    proba: np.ndarray,
    y_true: np.ndarray,
    cycles: list[str],
) -> IsotonicRegression | None:
    """Fit isotonic calibration only when a later validation slice improves Brier."""
    ordered_cycles = list(dict.fromkeys(cycles))
    if len(ordered_cycles) < 4:
        return None
    midpoint = len(ordered_cycles) // 2
    fit_cycles = set(ordered_cycles[:midpoint])
    score_cycles = set(ordered_cycles[midpoint:])
    fit_mask = np.asarray([cycle in fit_cycles for cycle in cycles])
    score_mask = np.asarray([cycle in score_cycles for cycle in cycles])
    if len(set(y_true[fit_mask].tolist())) < 2 or len(set(y_true[score_mask].tolist())) < 2:
        return None
    try:
        probe = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        probe.fit(proba[fit_mask], y_true[fit_mask])
        raw_brier = float(brier_score_loss(y_true[score_mask], proba[score_mask]))
        calibrated_brier = float(
            brier_score_loss(y_true[score_mask], probe.predict(proba[score_mask]))
        )
    except ValueError:
        logger.warning("Skipping isotonic calibration because validation scores could not be fit")
        return None
    if calibrated_brier >= raw_brier - 1e-6:
        return None
    calibrated = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrated.fit(proba, y_true)
    logger.info(
        "Isotonic calibration kept: holdout Brier improved from %.4f to %.4f",
        raw_brier,
        calibrated_brier,
    )
    return calibrated


def _labels(frame: pd.DataFrame) -> np.ndarray:
    if "actual_shortfall_at_cutoff" not in frame.columns:
        raise ModelError("Snapshots must include actual_shortfall_at_cutoff as the training target")
    return frame["actual_shortfall_at_cutoff"].to_numpy(dtype=int)


def _fit_estimator(
    kind: EstimatorKind, features: pd.DataFrame, y_true: np.ndarray, seed: int
) -> object:
    if kind == EstimatorKind.RULES:
        return LiquidityGapBaseline().fit(features, y_true)
    if kind == EstimatorKind.LOGISTIC:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipeline.fit(features, y_true)
        return pipeline
    if kind == EstimatorKind.XGBOOST:
        positives = max(int(np.sum(y_true)), 1)
        negatives = int(len(y_true) - np.sum(y_true))
        classifier = XGBClassifier(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
            tree_method="exact",
            scale_pos_weight=float(negatives) / float(positives),
            verbosity=0,
        )
        classifier.fit(features, y_true)
        return classifier
    _assert_never(kind)


def _importances(kind: EstimatorKind, estimator: object) -> dict[str, float] | None:
    if kind == EstimatorKind.LOGISTIC:
        if not isinstance(estimator, Pipeline):
            raise ModelError("Expected a logistic pipeline")
        coefficients = estimator.named_steps["clf"].coef_.ravel()
        return {
            name: float(value) for name, value in zip(MODEL_FEATURES, coefficients, strict=True)
        }
    if kind == EstimatorKind.XGBOOST:
        if not isinstance(estimator, XGBClassifier):
            raise ModelError("Expected an XGBoost classifier")
        return {
            name: float(value)
            for name, value in zip(MODEL_FEATURES, estimator.feature_importances_, strict=True)
        }
    if kind == EstimatorKind.RULES:
        return None
    _assert_never(kind)


def _explanation(kind: EstimatorKind, calibrated: bool) -> str:
    calibration = (
        " Isotonic calibration was kept because it improved holdout Brier score."
        if calibrated
        else ""
    )
    if kind == EstimatorKind.RULES:
        return (
            "The baseline maps the gap between forecast net debit and prediction-time "
            "liquidity through a logistic curve. The scale is the training-set median gap."
            + calibration
        )
    if kind == EstimatorKind.LOGISTIC:
        return (
            "Logistic regression uses standardized features. Coefficients are associations "
            "on that scale, not causal effects." + calibration
        )
    if kind == EstimatorKind.XGBOOST:
        return (
            "XGBoost gain importances describe splits in the simulated sample. "
            "They are not a causal explanation." + calibration
        )
    _assert_never(kind)


def _stress_slices(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
) -> dict[str, ClassificationMetrics]:
    if "scenario" not in frame.columns:
        return {}
    slices: dict[str, ClassificationMetrics] = {}
    scenarios = frame["scenario"].to_numpy()
    for scenario in sorted(set(scenarios.tolist())):
        mask = scenarios == scenario
        slices[str(scenario)] = _classification_metrics(
            y_true[mask],
            proba[mask],
            threshold,
            split=f"test:{scenario}",
        )
    return slices


def _evaluate_kind(
    kind: EstimatorKind,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> tuple[ModelRun, object, SavedShortfallModel]:
    x_train = feature_matrix(train)
    y_train = _labels(train)
    estimator = _fit_estimator(kind, x_train, y_train, seed)
    raw_validation = _positive_probability(kind, estimator, feature_matrix(validation))
    isotonic = _maybe_calibrate(
        raw_validation, _labels(validation), validation["cycle_id"].tolist()
    )
    validation_proba = (
        raw_validation
        if isotonic is None
        else np.asarray(isotonic.predict(raw_validation), dtype=float)
    )
    validation_proba = np.clip(validation_proba, 0.0, 1.0)
    threshold = select_threshold(_labels(validation), validation_proba)
    validation_metrics = _classification_metrics(
        _labels(validation),
        validation_proba,
        threshold,
        split="validation",
    )
    raw_test = _positive_probability(kind, estimator, feature_matrix(test))
    test_proba = (
        raw_test if isotonic is None else np.asarray(isotonic.predict(raw_test), dtype=float)
    )
    test_proba = np.clip(test_proba, 0.0, 1.0)
    test_metrics = _classification_metrics(_labels(test), test_proba, threshold, split="test")
    run = ModelRun(
        name=kind.value,
        threshold=threshold,
        calibrated=isotonic is not None,
        validation=validation_metrics,
        test=test_metrics,
        feature_importances=_importances(kind, estimator),
        explanation=_explanation(kind, isotonic is not None),
        stress_slices=_stress_slices(test, _labels(test), test_proba, threshold),
    )
    saved = SavedShortfallModel(
        model_version=f"scse-{kind.value}-v1",
        model_name=kind.value,
        feature_names=list(MODEL_FEATURES),
        threshold=threshold,
        seed=seed,
        estimator_kind=kind,
        estimator=estimator,
        isotonic=isotonic,
    )
    return run, estimator, saved


def train_and_evaluate(
    snapshots: pd.DataFrame,
    *,
    seed: int = 42,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
) -> TrainingResult:
    """Fit all three approaches and score the untouched test cycles once.

    Model selection uses validation F1. On a tie, the simpler approach is kept.
    """
    train, validation, test = chronological_cycle_split(
        snapshots,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    _require_both_classes(_labels(train), "training", minimum_positives=3)
    _require_both_classes(_labels(validation), "validation", minimum_positives=1)
    _require_both_classes(_labels(test), "test", minimum_positives=1)
    runs: list[ModelRun] = []
    estimators: dict[str, object] = {}
    saved_models: dict[str, SavedShortfallModel] = {}
    for kind in EstimatorKind:
        run, estimator, saved = _evaluate_kind(kind, train, validation, test, seed)
        runs.append(run)
        estimators[kind.value] = estimator
        saved_models[kind.value] = saved
        logger.info(
            "Evaluated %s validation_f1=%.3f test_f1=%.3f threshold=%.2f calibrated=%s",
            kind.value,
            run.validation.f1,
            run.test.f1,
            run.threshold,
            run.calibrated,
        )
    selected_run = sorted(
        runs, key=lambda run: (-run.validation.f1, _COMPLEXITY[EstimatorKind(run.name)], run.name)
    )[0]
    report = ComparisonReport(
        runs=tuple(runs),
        selected_model=selected_run.name,
        seed=seed,
        train_cycles=tuple(dict.fromkeys(train["cycle_id"].tolist())),
        validation_cycles=tuple(dict.fromkeys(validation["cycle_id"].tolist())),
        test_cycles=tuple(dict.fromkeys(test["cycle_id"].tolist())),
        train_positive_rate=float(np.mean(_labels(train))),
        test_positive_rate=float(np.mean(_labels(test))),
        limitation=LIMITATION,
    )
    return TrainingResult(
        report=report, model=saved_models[selected_run.name], estimators=estimators
    )


def save_model(model: SavedShortfallModel, path: Path) -> None:
    """Persist a predictor. The file is an artifact, not a source of truth for obligations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved shortfall model %s to %s", model.model_version, path)


def load_model(path: Path) -> SavedShortfallModel:
    """Load a predictor and refuse artifacts trained on a different feature contract."""
    loaded = joblib.load(path)
    if not isinstance(loaded, SavedShortfallModel):
        raise ModelError(f"{path} does not contain a SavedShortfallModel")
    if list(loaded.feature_names) != list(MODEL_FEATURES):
        raise ModelError("Saved model feature names do not match the current contract")
    return loaded


def score_snapshots(model: SavedShortfallModel, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Score rows without reading the label column."""
    probabilities = model.predict_proba_positive(snapshots)
    decisions = (probabilities >= model.threshold).astype(int)
    return pd.DataFrame(
        {
            "cycle_id": snapshots["cycle_id"].to_numpy(),
            "bank_id": snapshots["bank_id"].to_numpy(),
            "probability": probabilities,
            "predicted_label": decisions,
            "model_version": model.model_version,
            "threshold": model.threshold,
        }
    )


def render_results_markdown(report: ComparisonReport) -> str:
    """Render the comparison that ``train_and_evaluate`` actually produced."""
    lines = [
        "# Experimental results",
        "",
        "These numbers were written by `scripts/run_evaluation.py` from a model run.",
        "They measure the synthetic simulator only.",
        "",
        report.limitation,
        "",
        f"Selected model (highest validation F1, then the simpler model): `{report.selected_model}`.",
        f"Seed: `{report.seed}`.",
        f"Train cycles: {len(report.train_cycles)}. Validation cycles: {len(report.validation_cycles)}. Test cycles: {len(report.test_cycles)}.",
        f"Training shortfall rate: {report.train_positive_rate:.3f}. Test shortfall rate: {report.test_positive_rate:.3f}.",
        "",
        "| Approach | Threshold | Calibrated | PR-AUC | Recall | Precision | F1 | Brier | TP | FP | FN | TN |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in report.runs:
        metrics = run.test
        average_precision = (
            "n/a" if metrics.average_precision is None else f"{metrics.average_precision:.3f}"
        )
        lines.append(
            "| {name} | {threshold:.2f} | {calibrated} | {average_precision} | {recall:.3f} | "
            "{precision:.3f} | {f1:.3f} | {brier:.3f} | {tp} | {fp} | {fn} | {tn} |".format(
                name=run.name,
                threshold=run.threshold,
                calibrated="yes" if run.calibrated else "no",
                average_precision=average_precision,
                recall=metrics.recall,
                precision=metrics.precision,
                f1=metrics.f1,
                brier=metrics.brier,
                tp=metrics.true_positive,
                fp=metrics.false_positive,
                fn=metrics.false_negative,
                tn=metrics.true_negative,
            )
        )
    lines.extend(["", "## Validation metrics used for selection", ""])
    lines.append("| Approach | Validation F1 | Validation Brier |")
    lines.append("| --- | ---: | ---: |")
    for run in report.runs:
        lines.append(f"| {run.name} | {run.validation.f1:.3f} | {run.validation.brier:.3f} |")
    lines.extend(["", "## Stress slices on the test period", ""])
    for run in report.runs:
        lines.append(f"### {run.name}")
        lines.append("")
        lines.append(run.explanation)
        lines.append("")
        if not run.stress_slices:
            lines.append("No scenario column was available.")
            lines.append("")
            continue
        lines.append("| Scenario | Support+ | Recall | Precision | F1 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for scenario, metrics in run.stress_slices.items():
            lines.append(
                f"| {scenario} | {metrics.support_positive} | {metrics.recall:.3f} | "
                f"{metrics.precision:.3f} | {metrics.f1:.3f} |"
            )
        lines.append("")
    xgboost_run = next(
        (run for run in report.runs if run.name == EstimatorKind.XGBOOST.value), None
    )
    if xgboost_run is not None and xgboost_run.feature_importances:
        lines.extend(["## XGBoost gain (not causal)", ""])
        for name, value in sorted(
            xgboost_run.feature_importances.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            lines.append(f"- `{name}`: {value:.4f}")
        lines.append("")
    lines.append(
        "False positives are predicted shortfalls that did not occur. False negatives are missed shortfalls."
    )
    lines.append("")
    return "\n".join(lines)
