# Model card

## Summary

The model estimates whether a simulated bank's eligible net debit at the settlement cutoff will exceed the liquidity available at that cutoff.

- `1` means available liquidity will be insufficient.
- `0` means available liquidity will be sufficient.

This card describes a teaching model trained on a simulator. It is not a validated risk model for a bank, a card scheme, or a settlement system.

## Intended use

Compare a transparent baseline with two classifiers, inspect which pre-cutoff signals move the synthetic score, and feed those scores into a documented funding policy.

Intended users are reviewers of this portfolio project. The intended interface is the local API and the Streamlit page.

## Out of scope

- Real-time authorization, clearing files, or settlement instructions.
- Regulatory capital, liquidity coverage, or intraday credit decisions.
- Any claim that a score equals a future shortfall frequency.
- Causal claims from logistic coefficients or XGBoost gain.

SHAP is not part of this release. Gain and coefficients are associations in the simulated sample.

## Training data

Rows come from `smart_clearing.simulator`. Each row is one bank in one weekday cycle. The generator is seeded. Assumptions are listed by `synthetic_assumptions()` and copied into the experimental results file when `scripts/run_evaluation.py` runs.

The simulator is built so a demo run contains both shortfall and non-shortfall rows. That is a design choice, not a measurement of how often banks are short.

Target construction:

1. Keep records at or before the cutoff in the cycle currency.
2. Run the DNS engine.
3. Net debit is the positive amount the bank owes.
4. The label is 1 only when that debit is strictly greater than liquidity at the cutoff.

Liquidity at the cutoff can differ from liquidity at prediction time because of a simulated funding shock. The shock flag and the cutoff balance are not model features.

## Features

Only these columns are passed to the estimators:

- `incoming_volume_so_far`
- `outgoing_volume_so_far`
- `transaction_count_so_far`
- `average_transaction_amount_so_far`
- `recent_volume_change`
- `historical_shortfall_rate`
- `available_liquidity_at_prediction`
- `predicted_or_forecast_net_debit`

`feature_matrix` rejects label columns, cutoff liquidity, and scenario tags. `compute_observed_features` rejects any record timestamped after the prediction time.

## Evaluation

- Splits are chronological by `cycle_id`. Banks from the same cycle are not placed in different splits.
- The logistic regression pipeline standardizes features with a scaler fit on the training cycles. Class weight is balanced.
- XGBoost uses `scale_pos_weight` from the training class counts, `tree_method="exact"`, and a fixed seed.
- The baseline is a logistic curve of forecast debit minus prediction-time liquidity. Its scale is the training median absolute gap.
- Isotonic calibration is adopted only when it lowers Brier score on a later slice of the validation cycles. It is then refit on the full validation slice. Validation metrics are therefore not an unbiased test.
- The decision threshold maximizes validation F1. Ties keep the lower threshold.
- The reported comparison is one pass over the untouched test cycles.
- If the training, validation, or test split does not contain both classes, training raises `InsufficientDataError` and does not emit a metric.

The latest measured table is [experimental_results.md](experimental_results.md), produced by:

```bash
python scripts/run_evaluation.py --seed 42
```

Replace that file by rerunning the script. Do not paste metrics from memory.

## Decision policy

The selected artifact is the approach with the highest validation F1. If scores tie, the simpler approach is kept: rules baseline, then logistic regression, then XGBoost. Test-set F1 is not an input to that choice.

`artifacts/shortfall_model.joblib` stores the feature names, threshold, seed, estimator, and optional calibrator. `load_model` refuses a file whose feature list does not match the current contract. Prediction rejects a frame with missing columns or missing values.

## Failure modes

- The simulator can under-produce shortfalls if liquidity factors or the amount distribution change. The code stops instead of fabricating accuracy.
- A threshold chosen on a small validation slice can look strong on validation and weaker on the test period. Read the test columns.
- In-sample probabilities on the dashboard, for cycles that were in training, are not the test metrics.
- A funding shock after prediction time can create a shortfall the pre-cutoff features did not see. That is intentional label noise, and it is a reason the model can miss events.
- Class balance in the simulator is not the class balance of any real settlement system. Precision and recall will not transfer.
- Calibration on a few dozen positive rows is unstable. The report states whether isotonic calibration was kept.

## Ethical and operational note

Publishing a high F1 on this sample would not justify using the score to move liquidity in a real payment system. The dashboard and the API both state that recommendations are not executed payments.
