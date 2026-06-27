"""Feature giocatore — BLOCCO CARRIERA.

Per ogni (giocatore, stagione) calcola, usando SOLO il passato (no leakage):
  - SEASONS_EXP: anni di esperienza in lega (n. stagioni precedenti giocate);
  - SEASONS_WITH_TEAM: da quante stagioni consecutive e' in quella franchigia;
  - HOMETOWN_SCORE (0-4, tesi T1): scala graduata cresciuto-in-casa;
  - PLAYOFF_GAMES_PRIOR / PLAYOFF_DEPTH_PRIOR (tesi T3): esperienza playoff
    accumulata negli anni PRECEDENTI, pesata per profondita' raggiunta.

Note metodologiche:
  - TEAM_ID NBA e' stabile attraverso le rilocazioni di franchigia (Seattle ->
    OKC mantiene lo stesso id), quindi l'hometown si calcola direttamente su
    TEAM_ID senza mappare i trasferimenti citta'.
  - L'esperienza playoff usa playoff_series.parquet per sapere fino a che round
    e' arrivata la squadra del giocatore in ogni stagione passata.

Output (data/interim/):
  player_career.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

LONG_TENURE_MIN = 4  # stagioni nella squadra per contare "di lunga data" (hometown=2)

# Peso della profondita' playoff (per round raggiunto) per l'esperienza T3.
ROUND_DEPTH_WEIGHT = {1: 1, 2: 2, 3: 4, 4: 6}  # 1oT, semif, finali conf, Finals
TITLE_BONUS = 4  # bonus extra per chi ha vinto il titolo


def _load_draft_team() -> pd.Series:
    """PLAYER_ID -> TEAM_ID che lo ha draftato (solo draft NBA reali)."""
    dh = pd.read_parquet(RAW / "draft_history.parquet")
    dh = dh.dropna(subset=["PERSON_ID", "TEAM_ID"])
    dh = dh[dh["TEAM_ID"] > 0]
    # un giocatore appare una volta nel draft; in caso di duplicati teniamo il primo
    return dh.drop_duplicates("PERSON_ID").set_index("PERSON_ID")["TEAM_ID"]


def _team_playoff_depth() -> pd.DataFrame:
    """Per ogni (stagione, team_abbr) il round massimo raggiunto e se ha vinto.

    Lo deduciamo dalle serie: la squadra raggiunge il round R se compare in una
    serie di quel round; vince il titolo se e' WINNER delle Finals.
    """
    s = pd.read_parquet(INTERIM / "playoff_series.parquet")
    rows = []
    for (season, syear), grp in s.groupby(["SEASON", "SEASON_START_YEAR"]):
        # squadre presenti e round massimo a cui compaiono
        depth = {}
        champ = grp.loc[grp["ROUND"] == 4, "WINNER"]
        champ = champ.iloc[0] if len(champ) else None
        for _, r in grp.iterrows():
            for t in (r["TEAM_A"], r["TEAM_B"]):
                depth[t] = max(depth.get(t, 0), int(r["ROUND"]))
        for team, mx in depth.items():
            rows.append({
                "SEASON_START_YEAR": int(syear), "TEAM_ABBREVIATION": team,
                "MAX_ROUND": mx, "WON_TITLE": int(team == champ),
            })
    return pd.DataFrame(rows)


def build_player_career() -> pd.DataFrame:
    tech = pd.read_parquet(INTERIM / "player_technical_rs.parquet")
    draft_team = _load_draft_team()
    depth = _team_playoff_depth()

    # Una riga "principale" per (giocatore, stagione): se ha giocato in piu'
    # squadre, prendiamo quella con piu' minuti totali (il team "di stagione").
    tech = tech.sort_values("MIN_TOT", ascending=False)
    main = tech.drop_duplicates(["PLAYER_ID", "SEASON_START_YEAR"]).copy()
    main = main.sort_values(["PLAYER_ID", "SEASON_START_YEAR"]).reset_index(drop=True)

    # --- esperienza in lega e permanenza in squadra ---
    main["SEASONS_EXP"] = main.groupby("PLAYER_ID").cumcount()  # 0 = rookie
    # permanenza consecutiva nella stessa franchigia
    same_team = main["TEAM_ID"] != main.groupby("PLAYER_ID")["TEAM_ID"].shift()
    main["_team_change"] = same_team.fillna(True).astype(int)
    main["_streak_id"] = main.groupby("PLAYER_ID")["_team_change"].cumsum()
    main["SEASONS_WITH_TEAM"] = main.groupby(["PLAYER_ID", "_streak_id"]).cumcount() + 1

    # --- hometown score (T1) ---
    main["DRAFT_TEAM_ID"] = main["PLAYER_ID"].map(draft_team)
    main["DRAFTED_BY_CURRENT"] = (main["DRAFT_TEAM_ID"] == main["TEAM_ID"]).astype(int)
    # ha mai lasciato la squadra di draft e poi e' tornato?
    main["HOMETOWN_SCORE"] = main.apply(_hometown_score, axis=1)

    # --- esperienza playoff accumulata (T3), solo anni PRECEDENTI ---
    main = _add_prior_playoff_experience(main, depth)

    # tieni le colonne carriera (+ chiavi)
    keep = [
        "SEASON", "SEASON_START_YEAR", "PLAYER_ID", "PLAYER_NAME",
        "TEAM_ID", "TEAM_ABBREVIATION", "AGE",
        "SEASONS_EXP", "SEASONS_WITH_TEAM",
        "DRAFT_TEAM_ID", "DRAFTED_BY_CURRENT", "HOMETOWN_SCORE",
        "PLAYOFF_GAMES_PRIOR", "PLAYOFF_DEPTH_PRIOR", "DEEP_RUNS_PRIOR", "TITLES_PRIOR",
    ]
    out = main[keep].copy()
    out.to_parquet(INTERIM / "player_career.parquet", index=False)
    print(f"player_career: {len(out)} righe; hometown_score dist:\n{out['HOMETOWN_SCORE'].value_counts().sort_index().to_string()}")
    return out


def _hometown_score(row: pd.Series) -> int:
    """Scala 0-4 (vedi spec §3.3)."""
    drafted_here = row["DRAFTED_BY_CURRENT"] == 1
    tenure = row["SEASONS_WITH_TEAM"]
    if drafted_here:
        # 4 = draftato e (sempre) qui; 3 = draftato, via e tornato.
        # distinguiamo con la permanenza: se e' qui da quando e' rookie -> 4.
        # SEASONS_EXP == SEASONS_WITH_TEAM-1 significa mai andato via.
        if row["SEASONS_EXP"] == row["SEASONS_WITH_TEAM"] - 1:
            return 4
        return 3
    # non draftato qui
    if tenure >= LONG_TENURE_MIN:
        return 2
    if tenure >= 2:
        return 1
    return 0


def _load_playoff_minutes() -> pd.DataFrame:
    """Minuti e partite playoff EFFETTIVI per (giocatore, stagione).

    Cruciale (richiesta utente): l'esperienza playoff va pesata sui MINUTI
    realmente giocati. Una stella a 40 min x 13 gare vale piu' di un panchinaro
    a 2 min, anche se la squadra e' arrivata allo stesso round.
    """
    frames = []
    for yr in range(1996, 2026):
        season = f"{yr}-{str(yr+1)[-2:]}"
        p = RAW / "player_base_po_totals" / f"{season}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)[["PLAYER_ID", "MIN", "GP"]].copy()
        df = df.rename(columns={"MIN": "PO_MIN", "GP": "PO_GP"})
        df["SEASON_START_YEAR"] = yr
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _add_prior_playoff_experience(main: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    """Per ogni riga somma l'esperienza playoff degli anni PRECEDENTI del
    giocatore, pesata per profondita' del round E per i MINUTI EFFETTIVI."""
    depth_idx = depth.set_index(["SEASON_START_YEAR", "TEAM_ABBREVIATION"])

    # per ogni (giocatore-stagione) qual e' il round raggiunto quell'anno?
    def season_depth(syear, team):
        try:
            r = depth_idx.loc[(syear, team)]
            return int(r["MAX_ROUND"]), int(r["WON_TITLE"])
        except KeyError:
            return 0, 0

    rounds = []
    titles = []
    for _, row in main.iterrows():
        mr, wt = season_depth(int(row["SEASON_START_YEAR"]), row["TEAM_ABBREVIATION"])
        rounds.append(mr)
        titles.append(wt)
    main["_season_round"] = rounds
    main["_season_title"] = titles

    # minuti playoff effettivi del giocatore in quella run
    po_min = _load_playoff_minutes()
    main = main.merge(po_min[["PLAYER_ID", "SEASON_START_YEAR", "PO_MIN", "PO_GP"]],
                      on=["PLAYER_ID", "SEASON_START_YEAR"], how="left")
    main["PO_MIN"] = main["PO_MIN"].fillna(0.0)
    main["PO_GP"] = main["PO_GP"].fillna(0).astype(int)

    # peso base = profondita' del round (+ bonus titolo)
    base_w = main["_season_round"].map(ROUND_DEPTH_WEIGHT).fillna(0)
    base_w = base_w + main["_season_title"] * TITLE_BONUS
    # ESPERIENZA PESATA SUI MINUTI: profondita' x (minuti / 100) -> chi gioca
    # tanto in una run profonda accumula molto piu' credito.
    main["_season_depth_w"] = base_w * (main["PO_MIN"] / 100.0)
    # versione "presenza" (round della squadra, non pesata) per confronto/leggibilita'
    main["_season_depth_round"] = base_w
    # partite playoff EFFETTIVE giocate (non solo presenza)
    main["_po_games"] = main["PO_GP"]
    # "deep run": arrivato in finale conf E avendo giocato minuti veri (>=10/gara)
    deep_minutes = (main["PO_GP"] > 0) & (main["PO_MIN"] / main["PO_GP"].clip(lower=1) >= 10)
    main["_deep_run"] = ((main["_season_round"] >= 3) & deep_minutes).astype(int)

    g = main.groupby("PLAYER_ID")
    # cumulativi SHIFTATI di 1 (escludono la stagione corrente -> no leakage)
    # PLAYOFF_GAMES_PRIOR ora conta le partite playoff EFFETTIVE (non presenze)
    main["PLAYOFF_GAMES_PRIOR"] = g["_po_games"].apply(lambda s: s.shift().cumsum()).reset_index(level=0, drop=True)
    # PLAYOFF_DEPTH_PRIOR ora e' pesata sui MINUTI (float)
    main["PLAYOFF_DEPTH_PRIOR"] = g["_season_depth_w"].apply(lambda s: s.shift().cumsum()).reset_index(level=0, drop=True)
    main["DEEP_RUNS_PRIOR"] = g["_deep_run"].apply(lambda s: s.shift().cumsum()).reset_index(level=0, drop=True)
    main["TITLES_PRIOR"] = g["_season_title"].apply(lambda s: s.shift().cumsum()).reset_index(level=0, drop=True)
    main["PLAYOFF_GAMES_PRIOR"] = main["PLAYOFF_GAMES_PRIOR"].fillna(0).astype(int)
    main["DEEP_RUNS_PRIOR"] = main["DEEP_RUNS_PRIOR"].fillna(0).astype(int)
    main["TITLES_PRIOR"] = main["TITLES_PRIOR"].fillna(0).astype(int)
    main["PLAYOFF_DEPTH_PRIOR"] = main["PLAYOFF_DEPTH_PRIOR"].fillna(0.0).round(2)
    return main


if __name__ == "__main__":
    build_player_career()
