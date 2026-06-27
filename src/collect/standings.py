"""Collector: classifiche di RS -> conference + seed per squadra/stagione.

LeagueStandingsV3 fornisce, per ogni squadra e stagione: Conference (East/West),
PlayoffRank (il seed 1-15) e record. Ci serve per ricostruire il bracket in modo
PRECISO: finali di conference Est vs Est / Ovest vs Ovest, Finals Est vs Ovest.

Output cache (data/raw/): standings/{season}.parquet
Output consolidato (data/interim/): team_conference.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguestandingsv3

from src.collect.cache import cached_endpoint, season_str

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"


def collect_standings(start_year: int) -> pd.DataFrame:
    df = cached_endpoint(
        key=f"standings/{season_str(start_year)}",
        endpoint_cls=leaguestandingsv3.LeagueStandingsV3,
        params=dict(season=season_str(start_year), season_type="Regular Season", league_id="00"),
        frame_index=0,
    )
    out = df.copy()
    out["SEASON_START_YEAR"] = start_year
    return out


def build_team_conference(start: int = 1996, end: int = 2025) -> pd.DataFrame:
    # le standings danno conference/seed per TEAM_ID; le sigle a 3 lettere
    # (usate dalle serie) le prendiamo dalle team features via TEAM_ID.
    tf = pd.read_parquet(ROOT / "data" / "processed" / "team_season_features.parquet")
    abbr_map = tf[["SEASON_START_YEAR", "TEAM_ID", "TEAM_ABBREVIATION"]].drop_duplicates()

    frames = []
    for yr in range(start, end + 1):
        df = collect_standings(yr)
        part = pd.DataFrame({
            "SEASON_START_YEAR": df["SEASON_START_YEAR"],
            "TEAM_ID": df["TeamID"],
            "CONFERENCE": df["Conference"],
            "SEED": df["PlayoffRank"],
            "WINS": df["WINS"], "LOSSES": df["LOSSES"],
        })
        frames.append(part)
        print(f"  {season_str(yr)}: {len(df)} squadre, conf={sorted(df['Conference'].unique())}")
    out = pd.concat(frames, ignore_index=True)
    out = out.merge(abbr_map, on=["SEASON_START_YEAR", "TEAM_ID"], how="left")
    INTERIM.mkdir(parents=True, exist_ok=True)
    out.to_parquet(INTERIM / "team_conference.parquet", index=False)
    n_missing = out["TEAM_ABBREVIATION"].isna().sum()
    print(f"team_conference: {len(out)} righe su {out['SEASON_START_YEAR'].nunique()} stagioni "
          f"({n_missing} sigle mancanti)")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1996)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    build_team_conference(args.start, args.end)
