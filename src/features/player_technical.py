"""Feature giocatore — BLOCCO TECNICO (statistiche di gioco).

Costruisce UNA riga per (giocatore, stagione) con:
  - box score per-game (punti, rimbalzi, assist, tiri, percentuali);
  - shooting profile da 3 (volume + efficienza) -> tesi T4;
  - advanced (TS%, eFG%, USG%, NET/OFF/DEF rating, PIE, AST%, REB%) -> impatto;
  - totali stagionali (MIN_TOT, PTS_TOT...) per pesare i contributi a squadra.

Unisce regular season base+advanced+totals. Pensato per essere richiamato sia
in RS sia (riusando season_type) in PO.

Output (data/interim/):
  player_technical_rs.parquet   tutte le stagioni, regular season
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

# Colonne che teniamo dal BASE per-game (le rinominiamo senza suffissi rank).
_BASE_KEEP = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE",
    "GP", "W", "L", "W_PCT", "MIN",
    "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
    "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "PF", "PTS", "PLUS_MINUS",
]
# Colonne advanced di interesse (impatto, efficienza, uso).
_ADV_KEEP = [
    "PLAYER_ID", "TEAM_ID",
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "OREB_PCT", "DREB_PCT", "REB_PCT", "TM_TOV_PCT",
    "EFG_PCT", "TS_PCT", "USG_PCT", "PACE", "PIE", "POSS",
]
# Dai totals prendiamo i volumi stagionali per i pesi.
_TOT_KEEP = ["PLAYER_ID", "TEAM_ID", "GP", "MIN", "PTS", "FG3M", "FG3A"]


def _season_path(folder: str, season: str) -> Path:
    return RAW / folder / f"{season}.parquet"


def build_player_technical_season(start_year: int, season_type: str = "rs") -> pd.DataFrame:
    """Una riga per giocatore-stagione (un giocatore puo' avere piu' righe se
    cambiato squadra: nba_api le tiene separate per TEAM_ID, le manteniamo)."""
    season = f"{start_year}-{str(start_year+1)[-2:]}"

    base = pd.read_parquet(_season_path(f"player_base_{season_type}_pergame", season))
    adv = pd.read_parquet(_season_path(f"player_adv_{season_type}_pergame", season))
    tot = pd.read_parquet(_season_path(f"player_base_{season_type}_totals", season))

    base = base[[c for c in _BASE_KEEP if c in base.columns]].copy()
    adv = adv[[c for c in _ADV_KEEP if c in adv.columns]].copy()
    tot = tot[[c for c in _TOT_KEEP if c in tot.columns]].copy()
    tot = tot.rename(columns={
        "GP": "GP_TOT", "MIN": "MIN_TOT", "PTS": "PTS_TOT",
        "FG3M": "FG3M_TOT", "FG3A": "FG3A_TOT",
    })

    df = base.merge(adv, on=["PLAYER_ID", "TEAM_ID"], how="left")
    df = df.merge(tot, on=["PLAYER_ID", "TEAM_ID"], how="left")

    df.insert(0, "SEASON", season)
    df.insert(1, "SEASON_START_YEAR", start_year)

    # --- Derivate utili per la tesi T4 (tiro da 3) ---
    # frequenza di tiro da 3 sul totale tentativi
    df["FG3A_RATE"] = (df["FG3A"] / df["FGA"]).where(df["FGA"] > 0)
    # punti da 3 sul totale punti (quanto un giocatore "vive" di triple)
    df["PTS_FROM_3_SHARE"] = (3 * df["FG3M"] / df["PTS"]).where(df["PTS"] > 0)

    return df


def build_all_player_technical(start: int, end: int, season_type: str = "rs") -> pd.DataFrame:
    frames = []
    for yr in range(start, end + 1):
        season = f"{yr}-{str(yr+1)[-2:]}"
        if not _season_path(f"player_base_{season_type}_pergame", season).exists():
            continue
        frames.append(build_player_technical_season(yr, season_type))
    out = pd.concat(frames, ignore_index=True)
    suffix = season_type
    out.to_parquet(INTERIM / f"player_technical_{suffix}.parquet", index=False)
    print(
        f"player_technical_{suffix}: {len(out)} righe giocatore-stagione "
        f"su {out['SEASON'].nunique()} stagioni, {out['PLAYER_ID'].nunique()} giocatori unici"
    )
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1996)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    build_all_player_technical(args.start, args.end, "rs")
    build_all_player_technical(args.start, args.end, "po")
