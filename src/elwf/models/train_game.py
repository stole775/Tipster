from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


@dataclass
class GameModelResult:
    predictions: pd.DataFrame
    metrics: dict[str, float]
    per_round: pd.DataFrame | None = None
    upcoming: pd.DataFrame | None = None


class GameWinnerModel:
    """Simple logistic regression for home win probability."""

    def __init__(self, max_iter: int = 2000) -> None:
        self.model = LogisticRegression(max_iter=max_iter)
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame, target_col: str = "y_home_win") -> None:
        self.feature_cols = [c for c in df.columns if c.startswith("f_")]
        if target_col not in df.columns:
            raise ValueError(f"Target column {target_col} missing from training data")
        self.model.fit(df[self.feature_cols], df[target_col])

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        if not self.feature_cols:
            raise RuntimeError("Model not fitted")
        return pd.Series(self.model.predict_proba(df[self.feature_cols])[:, 1], index=df.index)


def walk_forward_games(
    games_df: pd.DataFrame,
    feature_cols: Iterable[str] | None = None,
    target_col: str = "y_home_win",
    round_col: str = "round",
    upcoming_df: pd.DataFrame | None = None,
    min_train_size: int = 20,
) -> GameModelResult:
    """Train and evaluate round-by-round to avoid look-ahead bias."""

    if feature_cols is None:
        feature_cols = [c for c in games_df.columns if c.startswith("f_")]

    df = games_df.sort_values(["season", round_col, "game_code"]).reset_index(drop=True)
    preds = []
    per_round_records: list[dict[str, float | int]] = []
    last_model: LogisticRegression | None = None

    for r in sorted(df[round_col].unique()):
        train = df[df[round_col] < r]
        test = df[df[round_col] == r]
        if len(train) < min_train_size:
            per_round_records.append(
                {
                    "round": r,
                    "n_train": len(train),
                    "n_test": len(test),
                    "skipped": True,
                    "reason": f"min_train_size={min_train_size}",
                }
            )
            continue

        model = LogisticRegression(max_iter=2000)
        model.fit(train[list(feature_cols)], train[target_col])
        last_model = model

        proba = model.predict_proba(test[list(feature_cols)])[:, 1]
        out = test[["season", round_col, "game_code", target_col]].copy()
        out["p_home_win"] = proba
        preds.append(out)

        per_round_records.append(
            {
                "round": r,
                "n_train": len(train),
                "n_test": len(test),
                "skipped": False,
                "reason": "",
                "logloss": log_loss(test[target_col], proba),
                "brier": brier_score_loss(test[target_col], proba),
                "acc@0.5": accuracy_score(test[target_col], (proba >= 0.5).astype(int)),
            }
        )

    pred_df = pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()

    metrics: dict[str, float] = {}
    if not pred_df.empty:
        metrics = {
            "logloss": log_loss(pred_df[target_col], pred_df["p_home_win"]),
            "brier": brier_score_loss(pred_df[target_col], pred_df["p_home_win"]),
            "acc@0.5": accuracy_score(
                pred_df[target_col], (pred_df["p_home_win"] >= 0.5).astype(int)
            ),
        }

    if pred_df.empty:
        per_round_df = pd.DataFrame(per_round_records) if per_round_records else None
        raise ValueError(
            "No predictions were produced. "
            f"Check that you have at least min_train_size={min_train_size} training rows before the first evaluated round. "
            "See per-round diagnostics for details."
        )

    upcoming_preds: pd.DataFrame | None = None
    if upcoming_df is not None and last_model is not None:
        upcoming_df = upcoming_df.copy()
        missing_cols = [c for c in feature_cols if c not in upcoming_df.columns]
        if missing_cols:
            raise ValueError(f"Upcoming games missing feature columns: {missing_cols}")
        proba = last_model.predict_proba(upcoming_df[list(feature_cols)])[:, 1]
        upcoming_preds = upcoming_df[["season", round_col, "game_code"]].copy()
        upcoming_preds["p_home_win"] = proba
        upcoming_preds["is_future"] = True

    per_round_df = pd.DataFrame(per_round_records) if per_round_records else None

    return GameModelResult(pred_df, metrics, per_round=per_round_df, upcoming=upcoming_preds)
