"""Prediction engine for the interactive app.

Wraps model training (walk-forward up to a given season) and bracket simulation
so the Streamlit app stays thin. Two entry points:

  - predict_season(year): use teams already in the historical database (1996-2025)
  - predict_from_json(payload): use a custom roster JSON uploaded by the user

Both return advancement probabilities (per team) and matchup probabilities.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.model.backtest_bracket import _fit_until
from src.model.bracket import build_conference_bracket
from src.model.interactive_bracket import InteractiveBracket
from src.model.season_report import advancement_probabilities, matchup_probabilities
from src.model.series_predictor import SeriesPredictor
from src.model.walkforward import _feature_cols

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"


def available_seasons() -> list[int]:
    s = pd.read_parquet(INTERIM / "playoff_series.parquet")
    return sorted(s["SEASON_START_YEAR"].unique())


def load_engine_for(year: int):
    """Train the model on everything before `year` and return a predictor."""
    feat = _feature_cols(pd.read_parquet(PROCESSED / "series_dataset.parquet"))
    model = _fit_until(year, feat)
    return SeriesPredictor(model, feat_order=feat)


def predict_season(year: int, n_sims: int = 5000) -> dict:
    predictor = load_engine_for(year)
    root = build_conference_bracket(year)
    adv = advancement_probabilities(root, predictor, year, n_sims=n_sims)
    mu = matchup_probabilities(root, predictor, year)
    return {"advancement": adv, "matchups": mu, "year": year}


def interactive_for(year: int) -> InteractiveBracket:
    """Interactive what-if bracket for a historical season (with overrides)."""
    return InteractiveBracket(load_engine_for(year), year)


def real_outcome(year: int) -> dict:
    """Real bracket results for the chosen season (for comparison)."""
    s = pd.read_parquet(INTERIM / "playoff_series.parquet")
    sy = s[s["SEASON_START_YEAR"] == year]
    champ = sy.loc[sy["ROUND"] == 4, "WINNER"]
    return {
        "champion": champ.iloc[0] if len(champ) else None,
        "series": sy[["ROUND", "ROUND_NAME", "TEAM_A", "TEAM_B", "WINNER"]].to_dict("records"),
    }


def team_strengths(year: int) -> pd.DataFrame:
    """Headline strengths per playoff team for the season (for display)."""
    tf = pd.read_parquet(PROCESSED / "team_season_features.parquet")
    tc = pd.read_parquet(INTERIM / "team_conference.parquet")
    yr = tf[tf["SEASON_START_YEAR"] == year].merge(
        tc[["SEASON_START_YEAR", "TEAM_ABBREVIATION", "CONFERENCE", "SEED"]],
        on=["SEASON_START_YEAR", "TEAM_ABBREVIATION"], how="left",
    )
    cols = ["TEAM_ABBREVIATION", "CONFERENCE", "SEED", "W", "L", "NET_RATING",
            "T3_playoff_depth", "T4_3p_pct", "T2_height", "n_superstars"]
    cols = [c for c in cols if c in yr.columns]
    return yr[yr["SEED"].notna()][cols].sort_values("NET_RATING", ascending=False)
