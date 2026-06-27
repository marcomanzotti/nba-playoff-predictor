"""Collector: statistiche squadra-stagione, partite RS (head-to-head) e playoff.

Per ogni stagione scarica, in cache:
  - team_base / team_adv: stat di squadra (box + advanced), Regular Season.
  - games_rs: tutte le righe-partita di Regular Season (2 righe per partita),
    da cui ricostruiamo il record head-to-head tra ogni coppia di squadre.
  - games_po: tutte le righe-partita dei Playoffs, da cui ricostruiamo le serie.

Output cache (data/raw/):
  team_base_rs/{season}.parquet
  team_adv_rs/{season}.parquet
  games_rs/{season}.parquet
  games_po/{season}.parquet

Uso:
    from src.collect.teams import collect_all_team_seasons
    collect_all_team_seasons(1996, 2025)
"""
from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, leaguegamefinder

from src.collect.cache import cached_endpoint, season_str


def collect_team_stats(start_year: int, measure: str = "Base") -> pd.DataFrame:
    abbr = "base" if measure == "Base" else "adv"
    key = f"team_{abbr}_rs/{season_str(start_year)}"
    df = cached_endpoint(
        key=key,
        endpoint_cls=leaguedashteamstats.LeagueDashTeamStats,
        params=dict(
            season=season_str(start_year),
            season_type_all_star="Regular Season",
            measure_type_detailed_defense=measure,
            per_mode_detailed="PerGame",
            league_id_nullable="00",
        ),
        frame_index=0,
    )
    df = df.copy()
    df.insert(0, "SEASON", season_str(start_year))
    df.insert(1, "SEASON_START_YEAR", start_year)
    return df


def collect_games(start_year: int, season_type: str = "Regular Season") -> pd.DataFrame:
    """Righe-partita (una per squadra per partita). MATCHUP+WL bastano per
    head-to-head (RS) e ricostruzione serie (Playoffs)."""
    abbr = "rs" if season_type == "Regular Season" else "po"
    key = f"games_{abbr}/{season_str(start_year)}"
    df = cached_endpoint(
        key=key,
        endpoint_cls=leaguegamefinder.LeagueGameFinder,
        params=dict(
            season_nullable=season_str(start_year),
            season_type_nullable=season_type,
            league_id_nullable="00",
        ),
        frame_index=0,
    )
    df = df.copy()
    df.insert(0, "SEASON", season_str(start_year))
    df.insert(1, "SEASON_START_YEAR", start_year)
    return df


def collect_all_team_seasons(start: int, end: int) -> None:
    seasons = list(range(start, end + 1))
    for yr in seasons:
        tb = collect_team_stats(yr, "Base")
        ta = collect_team_stats(yr, "Advanced")
        gr = collect_games(yr, "Regular Season")
        gp = collect_games(yr, "Playoffs")
        print(
            f"  {season_str(yr)}: team_base={len(tb)} team_adv={len(ta)} "
            f"games_rs={len(gr)} games_po={len(gp)}"
        )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    args = ap.parse_args()
    collect_all_team_seasons(args.start, args.end)
