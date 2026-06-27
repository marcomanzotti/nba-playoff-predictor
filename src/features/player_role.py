"""Assegnazione FASCIA DI RUOLO a ogni giocatore-stagione.

Tre fasce, per cogliere i matchup per ruolo (richiesta utente: 'sono fisico sui
lunghi ma piccolo solo tiratore' deve diventare visibile):

  - backcourt  (piccoli: PG/SG)
  - wing       (ali: SF/combo)
  - frontcourt (lunghi: PF/C)

Metodo robusto, valido per TUTTI i giocatori (non solo quelli con posizione
Combine): combiniamo altezza (proxy/reale, sempre presente) e stile di gioco
(AST% alto -> piccolo; REB%/BLK alto -> lungo). Usiamo la posizione Combine,
quando c'e', come segnale aggiuntivo.

Output (data/interim/):
  player_role.parquet   PLAYER_ID, SEASON_START_YEAR, ROLE_BAND
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"

BANDS = ["backcourt", "wing", "frontcourt"]


def _band_from_signals(height: float, ast_pct: float, reb_pct: float,
                       blk: float, h_lo: float, h_hi: float) -> str:
    """Punteggio 'grande vs piccolo' da altezza + stile -> fascia."""
    score = 0.0
    # altezza (segnale principale)
    if height >= h_hi:
        score += 1.5
    elif height <= h_lo:
        score -= 1.5
    # stile: tanti assist = piccolo; tanti rimbalzi/stoppate = lungo
    if pd.notna(ast_pct):
        score -= (ast_pct - 0.15) * 4    # AST% alto spinge verso piccolo
    if pd.notna(reb_pct):
        score += (reb_pct - 0.10) * 6    # REB% alto spinge verso lungo
    if pd.notna(blk):
        score += min(blk, 2.0) * 0.4     # stoppate -> lungo
    if score >= 0.8:
        return "frontcourt"
    if score <= -0.8:
        return "backcourt"
    return "wing"


def build_player_role() -> pd.DataFrame:
    tech = pd.read_parquet(INTERIM / "player_technical_rs.parquet")
    phys = pd.read_parquet(INTERIM / "player_physical.parquet")[["PLAYER_ID", "HEIGHT_WO_SHOES"]]

    df = tech.merge(phys, on="PLAYER_ID", how="left")
    # soglie altezza per terzili (fasce relative alla popolazione NBA)
    h_lo = df["HEIGHT_WO_SHOES"].quantile(0.33)
    h_hi = df["HEIGHT_WO_SHOES"].quantile(0.66)

    df["ROLE_BAND"] = [
        _band_from_signals(h, a, r, b, h_lo, h_hi)
        for h, a, r, b in zip(
            df["HEIGHT_WO_SHOES"], df.get("AST_PCT"), df.get("REB_PCT"), df.get("BLK")
        )
    ]
    out = df[["PLAYER_ID", "PLAYER_NAME", "SEASON_START_YEAR", "TEAM_ID", "ROLE_BAND"]].copy()
    out.to_parquet(INTERIM / "player_role.parquet", index=False)
    print("player_role: distribuzione fasce")
    print(out["ROLE_BAND"].value_counts().reindex(BANDS).to_string())
    return out


if __name__ == "__main__":
    build_player_role()
