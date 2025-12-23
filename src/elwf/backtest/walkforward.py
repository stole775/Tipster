from __future__ import annotations

import pandas as pd

from elwf.models.train_game import GameModelResult, walk_forward_games


def run_walkforward(
    games_df: pd.DataFrame,
    features_df: pd.DataFrame,
    target_col: str = "y_home_win",
    upcoming_df: pd.DataFrame | None = None,
    min_train_size: int = 20,
) -> GameModelResult:
    """Merge labels + features and run walk-forward evaluation."""

    merged = (
        games_df.merge(features_df, on="game_code", how="inner", validate="one_to_one")
        .sort_values(["season", "round", "game_code"])
        .reset_index(drop=True)
    )
    return walk_forward_games(
        merged, target_col=target_col, upcoming_df=upcoming_df, min_train_size=min_train_size
    )
