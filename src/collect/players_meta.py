"""Collector: Combine (fisico+atletismo), storia draft, anagrafica giocatori.

  - combine/{year}: misure Combine per anno solare dell'evento (dal 2000).
    Endpoint: DraftCombineStats(season_all_time='YYYY')  <-- NB parametro!
  - draft_history: una sola tabella con TUTTO lo storico draft (PERSON_ID +
    TEAM_ID di chi ha draftato). Base per la feature hometown (T1).
  - players_index: anagrafica statica (id, nome, attivo) per join robusti.

Output cache (data/raw/):
  combine/{year}.parquet
  draft_history.parquet
  players_index.parquet

Uso:
    from src.collect.players_meta import (
        collect_all_combine, collect_draft_history, collect_players_index
    )
"""
from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import draftcombinestats, drafthistory
from nba_api.stats.static import players as static_players

from src.collect.cache import RAW_DIR, cached_endpoint

# La Combine moderna (anthro+atletismo) parte dal 2000 (vedi audit Fase 0).
COMBINE_FIRST_YEAR = 2000


def collect_combine(event_year: int) -> pd.DataFrame:
    """Combine di un singolo anno solare (es. 2015 = draft class 2015)."""
    df = cached_endpoint(
        key=f"combine/{event_year}",
        endpoint_cls=draftcombinestats.DraftCombineStats,
        params=dict(season_all_time=str(event_year), league_id="00"),
        frame_index=0,
    )
    df = df.copy()
    df.insert(0, "COMBINE_YEAR", event_year)
    return df


def collect_all_combine(first: int = COMBINE_FIRST_YEAR, last: int = 2025) -> None:
    for yr in range(first, last + 1):
        df = collect_combine(yr)
        print(f"  combine {yr}: {len(df)} giocatori")


def collect_draft_history() -> pd.DataFrame:
    """Tutto lo storico draft in un colpo (PERSON_ID, TEAM_ID, pick, anno)."""
    df = cached_endpoint(
        key="draft_history",
        endpoint_cls=drafthistory.DraftHistory,
        params=dict(league_id="00"),
        frame_index=0,
    )
    print(f"  draft_history: {len(df)} righe ({df['SEASON'].min()}-{df['SEASON'].max()})")
    return df


def collect_players_index() -> pd.DataFrame:
    """Anagrafica statica giocatori (offline, nessuna chiamata di rete)."""
    path = RAW_DIR / "players_index.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = pd.DataFrame(static_players.get_players())
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  players_index: {len(df)} giocatori")
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--combine-first", type=int, default=COMBINE_FIRST_YEAR)
    ap.add_argument("--combine-last", type=int, default=2025)
    args = ap.parse_args()

    collect_players_index()
    collect_draft_history()
    collect_all_combine(args.combine_first, args.combine_last)
