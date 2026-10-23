"""Write a reproducible synthetic transaction file and snapshot file."""

from __future__ import annotations

import argparse
from pathlib import Path

from smart_clearing.config import configure_logging
from smart_clearing.simulator import SimulatorConfig, simulate, synthetic_assumptions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic clearing data. This is not real bank data."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cycles", type=int, default=18)
    parser.add_argument("--banks", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    configure_logging("INFO")
    result = simulate(
        SimulatorConfig(
            seed=args.seed,
            n_cycles=args.cycles,
            n_banks=args.banks,
            mean_transactions_per_cycle=28,
            min_transactions_per_cycle=12,
        )
    )
    args.output.mkdir(parents=True, exist_ok=True)
    transactions_path = args.output / "transactions.csv"
    snapshots_path = args.output / "snapshots.csv"
    result.transactions.to_csv(transactions_path, index=False)
    result.snapshots.to_csv(snapshots_path, index=False)
    assumptions_path = args.output / "assumptions.txt"
    assumptions_path.write_text("\n".join(synthetic_assumptions()) + "\n", encoding="utf-8")
    print(f"Wrote {len(result.transactions)} transactions to {transactions_path}")
    print(f"Wrote {len(result.snapshots)} snapshots to {snapshots_path}")
    print("These files are synthetic. Do not describe them as observed payment activity.")


if __name__ == "__main__":
    main()
