from __future__ import annotations

import pandas as pd

ROLLING_METRICS = [
    "points_for",
    "points_against",
    "off_rating",
    "def_rating",
    "pace",
    "efg_pct",
    "tov_pct",
    "orb_pct",
    "ftr",
]


def prepare_team_game_frame(raw_games: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw boxscore-level data into one row per team per game.

    Expected columns in ``raw_games``:
        - season: int
        - round: int
        - game_code: int
        - team_code: str
        - opponent_code: str
        - is_home: bool
        - tipoff: datetime64[ns]
        - metrics listed in ``ROLLING_METRICS``

    Returns:
        DataFrame indexed by team/game, sorted by tipoff for rolling calculations.
    """

    missing = [c for c in ["season", "round", "game_code", "team_code", "opponent_code", "tipoff"] if c not in raw_games.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = raw_games.copy()
    frame = frame.sort_values(["team_code", "tipoff", "game_code"]).reset_index(drop=True)
    return frame


def add_rolling_team_features(team_games: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Compute rolling averages for each team with a look-back window.

    Rolling metrics are shifted by 1 to avoid using information from the current
    game (prevents label leakage in the walk-forward regime).
    """

    team_games = prepare_team_game_frame(team_games)
    features = team_games.copy()

    for metric in ROLLING_METRICS:
        if metric not in features.columns:
            continue
        features[f"roll_{metric}"] = (
            features.groupby("team_code")[metric]
            .transform(lambda s: s.rolling(window, min_periods=1).mean().shift())
        )

    features["rest_days"] = (
        features.groupby("team_code")["tipoff"]
        .transform(lambda s: s.diff().dt.total_seconds() / 86_400)
    )

    return features


def build_game_features(team_games: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Construct pre-game features for home vs. away matchups.

    Args:
        team_games: one row per team per game with raw metrics.
        window: rolling window for form metrics.

    Returns:
        DataFrame keyed by game_code with feature columns prefixed by ``f_``.
    """

    feats = add_rolling_team_features(team_games, window=window)

    home = feats[feats["is_home"]].add_prefix("home_")
    away = feats[~feats["is_home"]].add_prefix("away_")

    merged = home.merge(
        away,
        left_on=["home_game_code"],
        right_on=["away_game_code"],
        suffixes=("_home", "_away"),
        validate="one_to_one",
    )

    feature_cols = []
    for metric in ROLLING_METRICS + ["rest_days"]:
        home_col = f"home_roll_{metric}" if metric != "rest_days" else "home_rest_days"
        away_col = f"away_roll_{metric}" if metric != "rest_days" else "away_rest_days"
        if home_col in merged.columns and away_col in merged.columns:
            diff_col = f"f_{metric}_diff"
            merged[diff_col] = merged[home_col] - merged[away_col]
            feature_cols.append(diff_col)

    result = merged[
        ["home_season", "home_round", "home_game_code"] + feature_cols
    ].rename(
        columns={
            "home_season": "season",
            "home_round": "round",
            "home_game_code": "game_code",
        }
    )
    return result.sort_values(["season", "round", "game_code"]).reset_index(drop=True)
