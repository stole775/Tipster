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
    args = parser.parse_args()

    games_df = pd.read_csv(args.games)
    features_df = pd.read_csv(args.features)

    result = run_walkforward(games_df, features_df)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(args.out, index=False)

    metrics_path = args.out.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result.metrics, indent=2))

    print(f"Saved predictions to {args.out}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

