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
    args = parser.parse_args()

    games_df = pd.read_csv(args.games)
    features_df = pd.read_csv(args.features)

    upcoming_df = pd.read_csv(args.upcoming) if args.upcoming else None

    result = run_walkforward(games_df, features_df, upcoming_df=upcoming_df)
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

    if result.upcoming is not None:
        upcoming_path = args.out.with_name(args.out.stem + ".upcoming.csv")
        result.upcoming.to_csv(upcoming_path, index=False)
        print(f"Saved upcoming predictions to {upcoming_path}")


if __name__ == "__main__":
    main()
