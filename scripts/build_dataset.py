"""Build the canonical pre-match feature dataset from the yearly ATP match files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from matchmind_v1.data import (
    FIRST_YEAR,
    LAST_YEAR,
    TARGET_COLUMN,
    assign_players,
    backward_date_steps,
    build_dataset,
    check_dataset,
    clean_matches,
    duplicate_identities,
    load_matches,
    missing_by_group,
    order_matches,
    retention_by_year,
    rows_after_a_later_date,
)

DEFAULT_INPUT = Path("data/all")
DEFAULT_OUTPUT = Path("data/processed/match_features.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--end-year", type=int, default=LAST_YEAR)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    raw = load_matches(args.input_dir, args.start_year, args.end_year)
    cleaned = clean_matches(raw)
    matches = assign_players(order_matches(cleaned), seed=args.seed)

    kept = 100 * len(cleaned) / len(raw)
    print(f"{len(raw):,} raw rows, {len(cleaned):,} kept ({kept:.2f}%)")
    print(retention_by_year(raw, cleaned).to_string())
    print("\nrows missing at least one required field, by group (groups overlap):")
    print(missing_by_group(raw).to_string())

    print(f"\nduplicate match identities: {len(duplicate_identities(cleaned))}")
    print("chronology            before sorting  after sorting")
    print(
        f"  backward date steps {backward_date_steps(cleaned['tourney_date']):>15,}"
        f"{backward_date_steps(matches['tourney_date']):>15,}"
    )
    print(
        f"  rows out of order   {rows_after_a_later_date(cleaned['tourney_date']):>15,}"
        f"{rows_after_a_later_date(matches['tourney_date']):>15,}"
    )

    dataset = build_dataset(matches)
    check_dataset(dataset)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    wins = int(dataset[TARGET_COLUMN].sum())
    print(f"\n{len(dataset):,} rows x {len(dataset.columns)} columns -> {args.output}")
    print(f"dates {dataset['tourney_date'].iloc[0]} to {dataset['tourney_date'].iloc[-1]}")
    print(f"player 1 won {wins:,} ({100 * wins / len(dataset):.2f}%)")
    print(f"player 2 won {len(dataset) - wins:,}")
    print(f"built in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
