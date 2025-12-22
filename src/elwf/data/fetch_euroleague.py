from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.game_metadata import GameMetadata
from euroleague_api.play_by_play_data import PlayByPlayData

from elwf.settings import EuroleagueSettings


def fetch_games(
    season: int,
    settings: EuroleagueSettings | None = None,
) -> pd.DataFrame:
    """Return game metadata for a season.

    The EuroLeague API returns a list of games with fields such as game_code,
    round, home team, away team, and tip-off time. We normalize the response
    into a DataFrame for downstream processing.
    """

    settings = settings or EuroleagueSettings()
    gm = GameMetadata(settings.competition_code)
    games = gm.get_game_metadata_season(season)
    return pd.DataFrame(games)


def fetch_boxscore(
    season: int,
    game_code: int,
    settings: EuroleagueSettings | None = None,
) -> dict[str, Any]:
    """Retrieve a single game's boxscore payload."""

    settings = settings or EuroleagueSettings()
    bs = BoxScoreData(settings.competition_code)
    return bs.get_game_boxscore(season, game_code)


def fetch_play_by_play(
    season: int,
    game_code: int,
    settings: EuroleagueSettings | None = None,
) -> dict[str, Any]:
    """Retrieve play-by-play events for a game."""

    settings = settings or EuroleagueSettings()
    pbp = PlayByPlayData(settings.competition_code)
    return pbp.get_game_play_by_play(season, game_code)


def cache_json(payload: dict[str, Any], path: Path) -> None:
    """Persist raw JSON responses for reproducibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_cached_json(path: Path) -> dict[str, Any]:
    """Load a cached JSON payload if it exists."""

    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def iter_games(
    seasons: Iterable[int],
    settings: EuroleagueSettings | None = None,
) -> pd.DataFrame:
    """Concatenate metadata from multiple seasons."""

    frames = [fetch_games(season, settings=settings) for season in seasons]
    return pd.concat(frames, ignore_index=True)

