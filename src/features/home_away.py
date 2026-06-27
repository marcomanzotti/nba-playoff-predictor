"""Split CASA / TRASFERTA + vantaggio campo (richiesta utente: fondamentale).

Il fattore campo e' tra i predittori piu' forti nei playoff: chi ha il
vantaggio campo gioca gara 1,2,5,7 in casa. Lo rendiamo esplicito.

L'informazione e' gia' nei dati grezzi (campo MATCHUP: 'vs.' = casa, '@' =
trasferta), quindi NON si riscarica nulla: si estrae e si aggrega.

Produciamo, per (squadra, stagione), separatamente per Regular Season e Playoff:
  - record casa / trasferta (W, L, win%);
  - punti segnati/subiti medi casa vs trasferta -> net rating casa/trasferta;
  - 'home_split' = quanto la squadra dipende dal fattore campo
    (win% casa - win% trasferta).

Output (data/interim/):
  home_away_rs.parquet
  home_away_po.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"


def _split_for_season(games: pd.DataFrame, season: str, syear: int) -> pd.DataFrame:
    g = games.copy()
    g["IS_HOME"] = g["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    g["WIN"] = (g["WL"] == "W").astype(int)
    g["TEAM"] = g["TEAM_ABBREVIATION"]
    # punti subiti = punti squadra - plus/minus
    if "PLUS_MINUS" in g.columns:
        g["PTS_ALLOWED"] = g["PTS"] - g["PLUS_MINUS"]
    else:
        g["PTS_ALLOWED"] = pd.NA

    rows = []
    for team, sub in g.groupby("TEAM"):
        home = sub[sub.IS_HOME == 1]
        away = sub[sub.IS_HOME == 0]
        rec = {
            "SEASON": season, "SEASON_START_YEAR": syear, "TEAM_ABBREVIATION": team,
            "HOME_W": int(home.WIN.sum()), "HOME_L": int((home.WIN == 0).sum()),
            "AWAY_W": int(away.WIN.sum()), "AWAY_L": int((away.WIN == 0).sum()),
            "HOME_WIN_PCT": float(home.WIN.mean()) if len(home) else float("nan"),
            "AWAY_WIN_PCT": float(away.WIN.mean()) if len(away) else float("nan"),
            "HOME_NET": float((home.PTS - home.PTS_ALLOWED).mean()) if len(home) else float("nan"),
            "AWAY_NET": float((away.PTS - away.PTS_ALLOWED).mean()) if len(away) else float("nan"),
        }
        rec["HOME_SPLIT"] = (rec["HOME_WIN_PCT"] - rec["AWAY_WIN_PCT"]
                             if pd.notna(rec["HOME_WIN_PCT"]) and pd.notna(rec["AWAY_WIN_PCT"])
                             else float("nan"))
        rows.append(rec)
    return pd.DataFrame(rows)


def build_home_away(season_type: str = "rs", start: int = 1996, end: int = 2025) -> pd.DataFrame:
    folder = f"games_{season_type}"
    frames = []
    for yr in range(start, end + 1):
        season = f"{yr}-{str(yr+1)[-2:]}"
        p = RAW / folder / f"{season}.parquet"
        if not p.exists():
            continue
        df = _split_for_season(pd.read_parquet(p), season, yr)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(INTERIM / f"home_away_{season_type}.parquet", index=False)
    print(f"home_away_{season_type}: {len(out)} righe squadra-stagione")
    return out


if __name__ == "__main__":
    build_home_away("rs")
    build_home_away("po")
