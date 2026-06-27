"""Export an existing season to the new-season JSON format.

Serves two purposes:
  1. produce a concrete example file (data/sample_new_season.json) so users know
     exactly what to upload;
  2. let us validate the ingestion pipeline against the normal pipeline (same
     season should give very similar predictions both ways).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"


def export_season(year: int, out_path: Path) -> dict:
    season = f"{year}-{str(year+1)[-2:]}"
    base = pd.read_parquet(RAW / f"player_base_rs_pergame/{season}.parquet")
    adv = pd.read_parquet(RAW / f"player_adv_rs_pergame/{season}.parquet")
    tot = pd.read_parquet(RAW / f"player_base_rs_totals/{season}.parquet")
    career = pd.read_parquet(INTERIM / "player_career.parquet")
    career = career[career["SEASON_START_YEAR"] == year]
    physical = pd.read_parquet(INTERIM / "player_physical.parquet")
    level = pd.read_parquet(INTERIM / "player_level.parquet")
    level = level[level["SEASON_START_YEAR"] == year]

    p = base.merge(adv[["PLAYER_ID", "TEAM_ID", "TS_PCT", "USG_PCT", "AST_PCT",
                        "REB_PCT", "OREB_PCT", "DREB_PCT", "NET_RATING", "PIE"]],
                   on=["PLAYER_ID", "TEAM_ID"], how="left")
    p = p.merge(tot[["PLAYER_ID", "TEAM_ID", "MIN", "PTS", "FG3M", "FG3A"]]
                .rename(columns={"MIN": "MIN_TOT", "PTS": "PTS_TOT",
                                 "FG3M": "FG3M_TOT", "FG3A": "FG3A_TOT"}),
                on=["PLAYER_ID", "TEAM_ID"], how="left")
    p = p.merge(career[["PLAYER_ID", "HOMETOWN_SCORE", "SEASONS_EXP",
                        "PLAYOFF_DEPTH_PRIOR", "DEEP_RUNS_PRIOR", "TITLES_PRIOR"]],
                on="PLAYER_ID", how="left")
    p = p.merge(physical[["PLAYER_ID", "HEIGHT_WO_SHOES", "WINGSPAN",
                          "MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME"]],
                on="PLAYER_ID", how="left")
    p = p.merge(level[["PLAYER_ID", "LEVEL"]], on="PLAYER_ID", how="left")

    players = p.where(pd.notna(p), None).to_dict("records")

    # teams
    tf = pd.read_parquet(PROCESSED / "team_season_features.parquet")
    tf = tf[tf["SEASON_START_YEAR"] == year]
    tc = pd.read_parquet(INTERIM / "team_conference.parquet")
    tc = tc[tc["SEASON_START_YEAR"] == year]
    teams = tf.merge(tc[["TEAM_ID", "CONFERENCE", "SEED"]], on="TEAM_ID", how="left")
    tcols = ["TEAM_ID", "TEAM_ABBREVIATION", "CONFERENCE", "SEED", "W", "L", "W_PCT",
             "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE", "TS_PCT", "EFG_PCT",
             "FG3M", "FG3A", "FG3_PCT", "PTS", "HOME_WIN_PCT", "AWAY_WIN_PCT"]
    teams = teams[[c for c in tcols if c in teams.columns]]
    teams_list = teams.where(pd.notna(teams), None).to_dict("records")

    # first-round matchups (the 8 real ones)
    s = pd.read_parquet(INTERIM / "playoff_series.parquet")
    r1 = s[(s["SEASON_START_YEAR"] == year) & (s["ROUND"] == 1)]
    first_round = [{"TEAM_A": r["TEAM_A"], "TEAM_B": r["TEAM_B"]} for _, r in r1.iterrows()]

    # head-to-head
    h2h = pd.read_parquet(INTERIM / "h2h_records.parquet")
    h2h = h2h[h2h["SEASON_START_YEAR"] == year]
    qualified = set(r1["TEAM_A"]) | set(r1["TEAM_B"])
    h2h = h2h[h2h["TEAM"].isin(qualified) & h2h["OPP"].isin(qualified)]
    head_to_head = [{"TEAM": r["TEAM"], "OPP": r["OPP"], "WINS": int(r["WINS"])}
                    for _, r in h2h.iterrows()]

    payload = {
        "season_start_year": year,
        "players": players,
        "teams": teams_list,
        "first_round": first_round,
        "head_to_head": head_to_head,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Exported {season}: {len(players)} players, {len(teams_list)} teams, "
          f"{len(first_round)} first-round series -> {out_path.name}")
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--out", default=str(ROOT / "data" / "sample_new_season.json"))
    args = ap.parse_args()
    export_season(args.year, Path(args.out))
