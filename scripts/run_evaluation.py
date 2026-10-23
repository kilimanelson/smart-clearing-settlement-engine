"""Fit the three shortfall approaches and write the measured comparison.

The numbers in ``docs/experimental_results.md`` come from this script. They are
synthetic-sample metrics, not validated banking performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smart_clearing.config import configure_logging
from smart_clearing.model import render_results_markdown, save_model, train_and_evaluate
from smart_clearing.simulator import experiment_config, simulate, synthetic_assumptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate shortfall models on a synthetic sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=Path("docs/experimental_results.md"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/shortfall_model.joblib"))
    args = parser.parse_args()
    configure_logging("INFO")
    config = experiment_config(seed=args.seed)
    result = simulate(config)
    trained = train_and_evaluate(result.snapshots, seed=config.seed)
    body = render_results_markdown(trained.report)
    assumptions = "\n".join(f"- {item}" for item in synthetic_assumptions())
    document = (
        f"{body}\n## Simulation settings\n\n"
        f"- Cycles: {config.n_cycles}\n"
        f"- Banks: {config.n_banks}\n"
        f"- Seed: {config.seed}\n"
        f"- Mean transactions per cycle: {config.mean_transactions_per_cycle}\n\n"
        f"## Assumptions\n\n{assumptions}\n"
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(document, encoding="utf-8")
    save_model(trained.model, args.model)
    selected = trained.report.selected_model
    print(f"Selected model on validation F1: {selected}")
    print(f"Wrote {args.report}")
    print(f"Wrote {args.model}")


if __name__ == "__main__":
    main()
