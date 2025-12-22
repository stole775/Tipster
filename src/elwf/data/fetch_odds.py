from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import requests

from elwf.settings import OddsApiSettings

ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"


def fetch_odds(settings: OddsApiSettings) -> pd.DataFrame:
    """Fetch odds for upcoming EuroLeague events.

    Args:
        settings: API and request parameters.

    Returns:
        DataFrame with one row per bookmaker per event, normalized for analysis.
    """

    params = {
        "apiKey": settings.api_key,
        "regions": settings.region,
        "markets": settings.market,
        "oddsFormat": settings.odds_format,
    }
    url = ODDS_API_URL.format(sport=settings.sport)
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()

    rows: list[dict[str, Any]] = []
    for event in data:
        for bookmaker in event.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                for outcome in market.get("outcomes", []):
                    rows.append(
                        {
                            "event_id": event.get("id"),
                            "commence_time": _parse_ts(event.get("commence_time")),
                            "home_team": event.get("home_team"),
                            "away_team": event.get("away_team"),
                            "bookmaker": bookmaker.get("title"),
                            "market": market.get("key"),
                            "outcome": outcome.get("name"),
                            "price": outcome.get("price"),
                            "point": outcome.get("point"),
                            "last_update": _parse_ts(bookmaker.get("last_update")),
                        }
                    )

    return pd.DataFrame(rows)


def implied_probability_decimal(price: float) -> float:
    """Convert decimal odds to implied probability (no vig removal)."""

    if price <= 1:
        raise ValueError("Decimal price must be > 1")
    return 1.0 / price


def _parse_ts(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

