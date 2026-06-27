"""Clustering #2 — LIVELLO / IMPATTO del giocatore (tesi T5).

Per ogni (giocatore, stagione) assegna una categoria di "livello":
  superstar > all_star > quality_starter > role_player > bench > bench_warmer

Serve a:
  (a) descrivere la qualita' del roster (quante stelle, quanti role player);
  (b) misurare il COINVOLGIMENTO DEI NON-SUPERSTAR (T5): quanta produzione
      viene da giocatori sotto il livello All-Star.

Metodo: combiniamo segnali di impatto (PIE, NET_RATING, USG, MIN) e volume
(PTS, minuti totali) in un punteggio composito, standardizzato PER STAGIONE
(cosi il livello e' relativo all'epoca, gestendo i cambi di gioco nel tempo).
Poi tagliamo in categorie ordinali con soglie su percentili di lega.

Nota: usiamo regole su percentili invece di KMeans puro perche' le categorie
qui sono intrinsecamente ORDINALI e vogliamo soglie stabili e interpretabili
nel tempo. (Un KMeans non garantisce ordinamento ne' stabilita' tra stagioni.)

Output (data/interim/):
  player_level.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"

LEVELS = ["bench_warmer", "bench", "role_player", "quality_starter", "all_star", "superstar"]

# Componenti del punteggio di impatto (z-score per stagione) e i loro pesi.
IMPACT_COMPONENTS = {
    "PIE": 1.5,          # player impact estimate (impatto complessivo)
    "PTS": 1.0,          # produzione offensiva
    "MIN": 1.0,          # fiducia dell'allenatore (minuti)
    "USG_PCT": 0.8,      # centralita' nell'attacco
    "NET_RATING": 0.6,   # impatto sul +/- per 100
    "MIN_TOT": 0.7,      # volume stagionale (durata/affidabilita')
}


def _zscore_by_season(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("SEASON_START_YEAR")[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def build_player_level() -> pd.DataFrame:
    tech = pd.read_parquet(INTERIM / "player_technical_rs.parquet")
    df = tech.copy()

    # filtro minimo: ignoriamo chi ha giocato pochissimo (rumore), ma li
    # teniamo come bench_warmer di default.
    df["_impact"] = 0.0
    for col, w in IMPACT_COMPONENTS.items():
        if col in df.columns:
            z = _zscore_by_season(df, col).fillna(0)
            df["_impact"] += w * z

    # rank percentile dell'impatto DENTRO la stagione
    df["_pct"] = df.groupby("SEASON_START_YEAR")["_impact"].rank(pct=True)

    # soglie ordinali su percentile di lega (tarabili)
    #   superstar       top 3%
    #   all_star        top 3-10%
    #   quality_starter 10-30%
    #   role_player     30-60%
    #   bench           60-85%
    #   bench_warmer    resto
    def to_level(p):
        if p >= 0.97: return "superstar"
        if p >= 0.90: return "all_star"
        if p >= 0.70: return "quality_starter"
        if p >= 0.40: return "role_player"
        if p >= 0.15: return "bench"
        return "bench_warmer"

    df["LEVEL"] = df["_pct"].apply(to_level)
    df["LEVEL_ORD"] = df["LEVEL"].map({lv: i for i, lv in enumerate(LEVELS)})

    out = df[[
        "SEASON", "SEASON_START_YEAR", "PLAYER_ID", "PLAYER_NAME",
        "TEAM_ID", "TEAM_ABBREVIATION", "MIN", "MIN_TOT", "PTS",
        "_impact", "_pct", "LEVEL", "LEVEL_ORD",
    ]].rename(columns={"_impact": "IMPACT_SCORE", "_pct": "IMPACT_PCT"})
    out.to_parquet(INTERIM / "player_level.parquet", index=False)
    print(f"player_level: {len(out)} righe. Distribuzione livelli:")
    print(out["LEVEL"].value_counts().reindex(LEVELS).to_string())
    return out


if __name__ == "__main__":
    build_player_level()
