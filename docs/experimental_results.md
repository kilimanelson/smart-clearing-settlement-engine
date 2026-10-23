# Experimental results

These numbers were written by `scripts/run_evaluation.py` from a model run.
They measure the synthetic simulator only.

Metrics describe a synthetic simulator, not a live card network. They are not validated banking performance and do not justify using this prototype for real settlement, capital, or liquidity decisions. Feature gain and logistic coefficients are associations in the simulated sample, not evidence of causality. SHAP was not added.

Selected model (highest validation F1, then the simpler model): `xgboost`.
Seed: `42`.
Train cycles: 21. Validation cycles: 7. Test cycles: 8.
Training shortfall rate: 0.230. Test shortfall rate: 0.292.

| Approach | Threshold | Calibrated | PR-AUC | Recall | Precision | F1 | Brier | TP | FP | FN | TN |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rules_baseline | 0.30 | yes | 0.792 | 0.643 | 0.900 | 0.750 | 0.099 | 9 | 1 | 5 | 33 |
| logistic_regression | 0.40 | no | 0.876 | 0.929 | 0.684 | 0.788 | 0.082 | 13 | 6 | 1 | 28 |
| xgboost | 0.90 | no | 0.833 | 0.429 | 0.857 | 0.571 | 0.116 | 6 | 1 | 8 | 33 |

## Validation metrics used for selection

| Approach | Validation F1 | Validation Brier |
| --- | ---: | ---: |
| rules_baseline | 0.900 | 0.036 |
| logistic_regression | 0.880 | 0.072 |
| xgboost | 0.952 | 0.083 |

## Stress slices on the test period

### rules_baseline

The baseline maps the gap between forecast net debit and prediction-time liquidity through a logistic curve. The scale is the training-set median gap. Isotonic calibration was kept because it improved holdout Brier score.

| Scenario | Support+ | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| funding_shock | 9 | 0.556 | 0.833 | 0.667 |
| volume_drop | 1 | 0.000 | 0.000 | 0.000 |
| volume_spike | 4 | 1.000 | 1.000 | 1.000 |

### logistic_regression

Logistic regression uses standardized features. Coefficients are associations on that scale, not causal effects.

| Scenario | Support+ | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| funding_shock | 9 | 0.889 | 0.571 | 0.696 |
| volume_drop | 1 | 1.000 | 1.000 | 1.000 |
| volume_spike | 4 | 1.000 | 1.000 | 1.000 |

### xgboost

XGBoost gain importances describe splits in the simulated sample. They are not a causal explanation.

| Scenario | Support+ | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| funding_shock | 9 | 0.444 | 0.800 | 0.571 |
| volume_drop | 1 | 0.000 | 0.000 | 0.000 |
| volume_spike | 4 | 0.500 | 1.000 | 0.667 |

## XGBoost gain (not causal)

- `predicted_or_forecast_net_debit`: 0.5154
- `incoming_volume_so_far`: 0.1265
- `outgoing_volume_so_far`: 0.1164
- `historical_shortfall_rate`: 0.0766
- `average_transaction_amount_so_far`: 0.0535
- `available_liquidity_at_prediction`: 0.0393
- `recent_volume_change`: 0.0375
- `transaction_count_so_far`: 0.0348

False positives are predicted shortfalls that did not occur. False negatives are missed shortfalls.

## Simulation settings

- Cycles: 36
- Banks: 6
- Seed: 42
- Mean transactions per cycle: 48

## Assumptions

- Participants are fictional bank ids. Nothing here is a real institution's flow.
- Each cycle is one weekday. The simulation clock is UTC, and the intraday curve uses that clock.
- A purchase creates an issuer-to-acquirer obligation. Issuer and acquirer are always different.
- Amounts are integer minor units drawn from a lognormal, then clipped.
- Pending records stay pending. They are visible in the transaction file and ineligible for DNS.
- A reversal is a separate record that references the original. The original purchase remains cleared.
- One settlement currency is cleared per cycle. A minority of records are tagged with a second currency and then excluded.
- Prediction time is 14:00 and cutoff is 17:00. Features cannot see later events.
- The intraday forecast scales observed eligible volume by a fixed prior. It is not fit on that cycle's future.
- Liquidity at prediction time is the expected lognormal amount times the mean transaction count times that bank's issuer weight, times a fixed liquidity factor and small noise. It does not use the realized transactions.
- A funding shock can reduce liquidity after prediction time and before cutoff. That shocked balance is a label input, not a feature.
- A shortfall means the eligible net debit at cutoff is strictly greater than liquidity at cutoff.
- Historical shortfall rate uses only earlier cycles. The current label is applied after the snapshot is built.
- Net debit of zero (a creditor or a flat bank) is not a shortfall.
- These settings are chosen so a demonstration run contains both shortfalls and non-shortfalls. They are not a calibrated model of a real market.
