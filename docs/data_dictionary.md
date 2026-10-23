# Data dictionary

Amounts are integer minor units. For the USD demo, the display helper uses exponent 2, so 100 minor units render as `1.00 USD`. The engines do not store major-unit floats. Averages in the snapshot table are the only floating amounts, and they are feature inputs rather than obligations.

Nothing in these tables is an extract from a real bank, scheme, or processor.

## Transaction

| Column | Meaning |
| --- | --- |
| `transaction_id` | Unique id inside a submission. Duplicates reject the batch. |
| `cycle_id` | Settlement cycle. Demo ids look like `CY-20240102` and sort in time order. |
| `issuer_bank` | Bank that owes the acquirer on a cleared purchase. |
| `acquirer_bank` | Bank that is owed on a cleared purchase. Must differ from the issuer for ingestion. |
| `amount_minor` | Positive integer minor units. Zero and negative values can be represented to the clearing engine, which quarantines them. |
| `currency` | Three-letter code. A cycle clears one currency. |
| `status` | `pending`, `cleared`, or `reversed`. |
| `event_timestamp` | Timezone-aware event time. |
| `clearing_timestamp` | Time the simulated status is known. Empty for pending records. |
| `reversal_of_transaction_id` | Required for `reversed`. Forbidden for other statuses at ingestion. |

### Clearing treatment

| Condition | Result |
| --- | --- |
| Duplicate id | Whole submission rejected |
| Amount ≤ 0 | Quarantined `non_positive_amount` |
| Issuer equals acquirer | Quarantined `same_bank` |
| Different cycle | Quarantined `wrong_cycle` |
| Different currency | Quarantined `wrong_currency` |
| Pending | Quarantined `not_eligible_status` |
| Cleared with a reversal link | Quarantined `cleared_has_reversal_reference` |
| Reversed without a parent id | Quarantined `reversal_missing_reference` |
| Cleared purchase | Issuer owes acquirer |
| Valid reversal | Acquirer owes issuer the reversal amount |

The first matching rule wins.

## Net position

| Column | Meaning |
| --- | --- |
| `bank_id` | Participant, including a listed bank with no included activity. |
| `net_position_minor` | Gross credit minus gross debit. Negative means the bank owes money. |
| `gross_debit_minor` | Obligations owed by the bank after bilateral netting. |
| `gross_credit_minor` | Obligations owed to the bank after bilateral netting. |
| `currency` | Cycle currency. |

Identity: the net positions of a report sum to 0.

## Snapshot

Built per bank and cycle. Model features are only the columns in `MODEL_FEATURES`.

| Column | Role | Definition |
| --- | --- | --- |
| `cycle_id` | Key | Cycle identifier. Not a model feature. |
| `bank_id` | Key | Participant. Not a model feature. |
| `prediction_timestamp` | Clock | 14:00 UTC in the demo. Features cannot use later events. |
| `cutoff_timestamp` | Clock | 17:00 UTC in the demo. Must be later than prediction. |
| `incoming_volume_so_far` | Feature | Gross credit from eligible records at or before prediction. |
| `outgoing_volume_so_far` | Feature | Gross debit from eligible records at or before prediction. |
| `transaction_count_so_far` | Feature | Count of positive-amount messages for the bank in the cycle currency at or before prediction, including pending. |
| `average_transaction_amount_so_far` | Feature | Mean minor units of that message count. Zero when the count is zero. |
| `recent_volume_change` | Feature | Last hour of message amount divided by the hour before it, minus one. Zero when both windows are empty. One when only the recent window has amount. |
| `historical_shortfall_rate` | Feature | Share of earlier cycles in which this bank's label was a shortfall. Zero when there is no history. |
| `available_liquidity_at_prediction` | Feature | Liquidity drawn from expected volume, issuer weight, a bank factor, and noise. Not from this cycle's realized payments. |
| `predicted_or_forecast_net_debit` | Feature | Positive debit implied by scaling the observed gross flows with a fixed intraday curve. |
| `actual_shortfall_at_cutoff` | Label | 1 when cutoff net debit exceeds cutoff liquidity. Not a feature. |
| `actual_final_net_debit` | Evaluation | Cutoff net debit, kept for retrospective checks. Not a feature. |
| `available_liquidity_at_cutoff` | Label input | Prediction liquidity, or a reduced balance when a post-prediction funding shock is drawn. Not a feature. |
| `scenario` | Slice | `normal`, `volume_spike`, `volume_drop`, or `funding_shock`. Not a feature. |
| `funding_shock_after_prediction` | Slice | Whether this bank's cutoff liquidity was reduced after prediction. Not a feature. |

## Database tables

PostgreSQL tables mirror the simulation: `banks`, `settlement_cycles`, `transactions`, `net_positions`, `liquidity_snapshots`, `risk_predictions`, `funding_recommendations`, `simulated_settlement_results`, and `audit_events`.

Transaction ids are primary keys. Positions, liquidity snapshots, and simulated results are unique per cycle and bank. Predictions are unique per cycle, bank, and model version. Recommendations are unique per cycle, bank, and strategy. Foreign keys point at cycles and banks. `payload_hash` detects a conflicting replay of the same transaction id.

`simulated_settlement_results.actual_shortfall` stays null at clear time. The service does not invent cutoff liquidity.

## Optimizer inputs

| Field | Meaning |
| --- | --- |
| `shortfall_probability` | Model score in `[0, 1]`. Not a guaranteed frequency. |
| `forecast_net_debit_minor` | Point debit the policy tries to cover. |
| `available_liquidity_minor` | Liquidity known at recommendation time. |
| `liquidity_cost_bps` | Cycle charge in basis points, from 0 to 10,000. |
| `funding_limit_minor` | Maximum additional funding for that bank. |
| `portfolio_funding_cap_minor` | Optional cap across banks. Applied by the risk-informed policy. |
| `risk_limit` | Probability at or above which the rules and risk-informed policies add a buffer. |

`additional_funding_minor` is a recommendation. It is not a settled payment.
