"""Predict a match between any two players in the saved history."""

from __future__ import annotations

import argparse
from pathlib import Path

from matchmind_v1.artifacts import load_bundle

DEFAULT_BUNDLE = Path("models/matchmind_v1_bundle.json.gz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-a", required=True, help="full name or numeric ATP id")
    parser.add_argument("--player-b", required=True, help="full name or numeric ATP id")
    parser.add_argument("--surface", default="Hard")
    parser.add_argument("--best-of", type=int, default=3)
    parser.add_argument("--draw-size", type=int, default=128)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    return parser.parse_args()


def as_date(stamp: int) -> str:
    text = str(stamp)
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def main() -> None:
    args = parse_args()
    if not args.bundle.exists():
        raise SystemExit(
            f"no prediction bundle at {args.bundle}; "
            "build one with scripts/build_prediction_bundle.py"
        )

    try:
        bundle = load_bundle(args.bundle)
        prediction = bundle.predict(
            args.player_a,
            args.player_b,
            surface=args.surface,
            best_of=args.best_of,
            draw_size=args.draw_size,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    a, b = prediction.player_a, prediction.player_b
    width = max(len(a.label), len(b.label)) + 1
    print(f"{a.label} vs {b.label}")
    print(f"Surface: {prediction.surface}")
    print(f"Best of: {prediction.best_of}")
    print()
    print(f"{a.label + ':':<{width}} {100 * prediction.probability_a:5.1f}%")
    print(f"{b.label + ':':<{width}} {100 * prediction.probability_b:5.1f}%")
    print()
    print(f"Predicted winner: {prediction.winner.label}")
    print()
    print(f"Historical data through: {as_date(prediction.history_end_date)}")
    print("Profiles last observed:")
    print(f"  {a.label}: {as_date(a.last_seen_date)}")
    print(f"  {b.label}: {as_date(b.last_seen_date)}")
    print()
    print("Probabilities are uncalibrated model outputs.")


if __name__ == "__main__":
    main()
