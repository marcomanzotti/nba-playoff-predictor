"""Collector: statistiche giocatore-stagione.

Per ogni stagione scarica, in cache:
  - Base    (box score) + Advanced, sia Regular Season sia Playoffs;
  - in PerGame e Totals (Totals serve per pesare correttamente i contributi,
    PerGame e' comodo per le feature).

Output cache (data/raw/):
  player_{measure}_{seasontype}_{permode}/{season}.parquet
  es. player_base_rs_pergame/2015-16.parquet

Uso:
    from src.collect.players import collect_player_season
    df = collect_player_season(2015, measure="Base", season_type="Regular Season")
"""
from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import leaguedashplayerstats

from src.collect.cache import cached_endpoint, season_str

# Abbreviazioni per i nomi dei file di cache.
_ST_ABBR = {"Regular Season": "rs", "Playoffs": "po"}
_PM_ABBR = {"PerGame": "pergame", "Totals": "totals", "Per100Possessions": "per100"}
_MT_ABBR = {"Base": "base", "Advanced": "adv"}


def collect_player_season(
    start_year: int,
    measure: str = "Base",
    season_type: str = "Regular Season",
    per_mode: str = "PerGame",
) -> pd.DataFrame:
    """Scarica (o legge da cache) una tabella giocatore-stagione."""
    key = (
        f"player_{_MT_ABBR[measure]}_{_ST_ABBR[season_type]}_{_PM_ABBR[per_mode]}"
        f"/{season_str(start_year)}"
    )
    df = cached_endpoint(
        key=key,
        endpoint_cls=leaguedashplayerstats.LeagueDashPlayerStats,
        params=dict(
            season=season_str(start_year),
            season_type_all_star=season_type,
            measure_type_detailed_defense=measure,
            per_mode_detailed=per_mode,
            league_id_nullable="00",
        ),
        frame_index=0,
    )
    # marchiamo stagione e contesto per quando concateneremo tutto
    df = df.copy()
    df.insert(0, "SEASON", season_str(start_year))
    df.insert(1, "SEASON_START_YEAR", start_year)
    return df


# Combinazioni che vogliamo per ogni stagione.
_PLAYER_GRID = [
    # (measure, season_type, per_mode)
    ("Base", "Regular Season", "PerGame"),
    ("Base", "Regular Season", "Totals"),
    ("Advanced", "Regular Season", "PerGame"),
    ("Base", "Playoffs", "PerGame"),
    ("Base", "Playoffs", "Totals"),
    ("Advanced", "Playoffs", "PerGame"),
]


def collect_all_player_seasons(start: int, end: int) -> None:
    """Scarica tutte le combinazioni per le stagioni [start, end] inclusi.

    Playoffs di stagioni senza playoff (improbabile nell'orizzonte) o tabelle
    vuote vengono comunque cache-ate come vuote dal layer di cache.
    """
    seasons = list(range(start, end + 1))
    total = len(seasons) * len(_PLAYER_GRID)
    done = 0
    for yr in seasons:
        for measure, st, pm in _PLAYER_GRID:
            df = collect_player_season(yr, measure=measure, season_type=st, per_mode=pm)
            done += 1
            print(f"  [{done:3d}/{total}] player {season_str(yr)} {measure}/{st}/{pm}: {len(df)} righe")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    args = ap.parse_args()
    collect_all_player_seasons(args.start, args.end)
