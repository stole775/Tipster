from __future__ import annotations

from pydantic import BaseModel, Field


class OddsApiSettings(BaseModel):
    """Configuration for The Odds API client."""

    api_key: str = Field(..., description="The Odds API key")
    region: str = Field("eu", description="Bookmaker region code, e.g., eu/us/uk/au")
    market: str = Field("h2h", description="Odds market to fetch (h2h/spreads/totals)")
    sport: str = Field(
        "basketball_euroleague",
        description="Sport key used by The Odds API for EuroLeague basketball.",
    )
    odds_format: str = Field(
        "decimal",
        description="Returned odds format (decimal, american).",
    )


class EuroleagueSettings(BaseModel):
    """Configuration for EuroLeague data fetchers."""

    competition_code: str = Field(
        "E", description="EuroLeague competition code used by the API (E for EuroLeague)."
    )

