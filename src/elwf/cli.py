from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from elwf.backtest.walkforward import run_walkforward


def main() -> None:
    parser = argparse.ArgumentParser(description="EuroLeague walk-forward toolkit")
    parser.add_argument("games", type=Path, help="CSV with columns: season, round, game_code, y_home_win")
    parser.add_argument("features", type=Path, help="CSV with features keyed by game_code")
    parser.add_argument("--out", type=Path, default=Path("artifacts/walkforward.csv"))
    parser.add_argument("--upcoming", type=Path, help="Optional CSV with upcoming games and features")
    parser.add_argument(
        "--min-train",
        type=int,
        default=20,
        help="Minimum number of training rows required before scoring a round (default: 20)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print per-round diagnostics even when predictions are empty or skipped",
    )
    args = parser.parse_args()

    def require_file(path: Path, label: str) -> None:
        if not path.exists():
            parser.error(
                f"Missing {label} file at '{path}'. "
                "Verify you are running the command from the project root and that the path is correct."
            )

    require_file(args.games, "games")
    require_file(args.features, "features")
    if args.upcoming is not None:
        require_file(args.upcoming, "upcoming")

    games_df = pd.read_csv(args.games)
    features_df = pd.read_csv(args.features)

    upcoming_df = pd.read_csv(args.upcoming) if args.upcoming else None

    result = run_walkforward(
        games_df,
        features_df,
        upcoming_df=upcoming_df,
        min_train_size=args.min_train,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.out, index=False)

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result.metrics, indent=2))
    print(f"Saved predictions to {args.out}")
    print(f"Saved metrics to {metrics_path}")

    if result.per_round is not None:
        per_round_path = args.out.with_suffix(".per_round.csv")
        result.per_round.to_csv(per_round_path, index=False)
        print(f"Saved per-round metrics to {per_round_path}")
        if args.debug:
            print(result.per_round.to_string(index=False))

    if result.upcoming is not None:
        upcoming_path = args.out.with_name(args.out.stem + ".upcoming.csv")
        result.upcoming.to_csv(upcoming_path, index=False)
        print(f"Saved upcoming predictions to {upcoming_path}")


if __name__ == "__main__":
    main()
