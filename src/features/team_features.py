"""Aggregazione a livello SQUADRA-STAGIONE + le 5 feature-tesi (T1-T5).

Unisce i blocchi giocatore (tecnico, carriera, fisico, livello) e li aggrega
per (squadra, stagione) pesando per i MINUTI TOTALI, poi aggiunge le team
stats ufficiali (record, rating, pace, tiro da 3).

Le 5 feature-tesi (spec §0):
  T1 HOMETOWN     -> hometown share pesato per minuti (quanta produzione viene
                     da giocatori cresciuti in casa)
  T2 FISICO/ATL   -> indici aggregati taglia + atletismo pesati per minuti
  T3 PLAYOFF EXP  -> esperienza playoff pesata per profondita', somma roster
  T4 TIRO DA 3    -> volume + efficienza da 3 a livello squadra
  T5 NON-SUPERSTAR-> coinvolgimento oltre le stelle (quota minuti/punti dei
                     non-superstar + entropia della distribuzione del contributo)

Output (data/processed/):
  team_season_features.parquet   una riga per (squadra, stagione)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# misure fisiche per gli indici T2
SIZE_COLS = ["HEIGHT_WO_SHOES", "WINGSPAN", "STANDING_REACH", "WEIGHT"]
ATHL_COLS = ["MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT"]


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    w = weights.fillna(0).clip(lower=0)
    v = values.astype(float)
    mask = v.notna() & (w > 0)
    if mask.sum() == 0 or w[mask].sum() == 0:
        return np.nan
    return float(np.average(v[mask], weights=w[mask]))


def _shannon_entropy(shares: np.ndarray) -> float:
    """Entropia normalizzata [0,1] della distribuzione del contributo (T5)."""
    p = shares[shares > 0]
    if len(p) <= 1:
        return 0.0
    h = -(p * np.log(p)).sum()
    return float(h / np.log(len(p)))


def _role_band_features(g: pd.DataFrame, w: str) -> dict:
    """Feature di squadra calcolate SEPARATAMENTE per fascia di ruolo.

    Per backcourt/wing/frontcourt, pesando sui minuti DENTRO la fascia:
      - quota minuti della fascia (quanto la squadra investe in quel ruolo)
      - fisico (altezza, wingspan, vertical) di quel reparto
      - tiro da 3 (% e volume) di quel reparto
      - esperienza playoff e livello medio di quel reparto
    Cosi 'lunghi fisici + piccoli tiratori' diventa esplicito.
    """
    out = {}
    total_min = g[w].sum()
    for band in ("backcourt", "wing", "frontcourt"):
        sub = g[g["ROLE_BAND"] == band]
        pref = f"BAND_{band[:4]}"  # back/wing/fron
        if len(sub) == 0 or sub[w].sum() <= 0:
            # reparto assente: valori neutri (0 quota, NaN per le medie)
            out[f"{pref}_minshare"] = 0.0
            for k in ("height", "wingspan", "vertical", "3p_pct", "3pa", "po_depth", "level"):
                out[f"{pref}_{k}"] = np.nan
            continue
        out[f"{pref}_minshare"] = float(sub[w].sum() / total_min)
        out[f"{pref}_height"] = _wmean(sub["HEIGHT_WO_SHOES"], sub[w])
        out[f"{pref}_wingspan"] = _wmean(sub["WINGSPAN"], sub[w])
        out[f"{pref}_vertical"] = _wmean(sub["MAX_VERTICAL_LEAP"], sub[w])
        out[f"{pref}_3p_pct"] = _wmean(sub["FG3_PCT"], sub[w])
        out[f"{pref}_3pa"] = _wmean(sub["FG3A"], sub[w])
        out[f"{pref}_po_depth"] = _wmean(sub["PLAYOFF_DEPTH_PRIOR"], sub[w])
        out[f"{pref}_level"] = _wmean(sub["LEVEL_ORD"], sub[w])
    return out


def build_team_features() -> pd.DataFrame:
    tech = pd.read_parquet(INTERIM / "player_technical_rs.parquet")
    career = pd.read_parquet(INTERIM / "player_career.parquet")
    physical = pd.read_parquet(INTERIM / "player_physical.parquet")
    level = pd.read_parquet(INTERIM / "player_level.parquet")

    # tech ha una riga per (player, season, team). Uniamo i blocchi.
    df = tech.copy()
    df = df.merge(
        career[["PLAYER_ID", "SEASON_START_YEAR", "HOMETOWN_SCORE",
                "PLAYOFF_DEPTH_PRIOR", "DEEP_RUNS_PRIOR", "TITLES_PRIOR", "SEASONS_EXP"]],
        on=["PLAYER_ID", "SEASON_START_YEAR"], how="left",
    )
    df = df.merge(physical[["PLAYER_ID"] + SIZE_COLS + ATHL_COLS + ["ARCHETYPE", "is_imputed"]],
                  on="PLAYER_ID", how="left")
    df = df.merge(level[["PLAYER_ID", "SEASON_START_YEAR", "LEVEL", "LEVEL_ORD"]],
                  on=["PLAYER_ID", "SEASON_START_YEAR"], how="left")
    role = pd.read_parquet(INTERIM / "player_role.parquet")
    df = df.merge(role[["PLAYER_ID", "SEASON_START_YEAR", "TEAM_ID", "ROLE_BAND"]],
                  on=["PLAYER_ID", "SEASON_START_YEAR", "TEAM_ID"], how="left")
    df["ROLE_BAND"] = df["ROLE_BAND"].fillna("wing")

    w = "MIN_TOT"  # peso = minuti totali stagionali
    rows = []
    for (syear, team_id), g in df.groupby(["SEASON_START_YEAR", "TEAM_ID"]):
        total_min = g[w].sum()
        if total_min <= 0:
            continue
        min_share = g[w] / total_min
        pts_tot = g["PTS_TOT"].fillna(0)
        pts_share = pts_tot / pts_tot.sum() if pts_tot.sum() > 0 else min_share

        # T1 hometown: quota minuti pesata dallo score (0-4) normalizzato
        hs = g["HOMETOWN_SCORE"].fillna(0)
        t1_hometown_minshare = float((min_share * (hs / 4.0)).sum())
        t1_homegrown_core = float((min_share * (hs >= 3).astype(int)).sum())  # quota min da draftati

        # T2 fisico/atletismo (pesati per minuti)
        t2_height = _wmean(g["HEIGHT_WO_SHOES"], g[w])
        t2_wingspan = _wmean(g["WINGSPAN"], g[w])
        t2_reach = _wmean(g["STANDING_REACH"], g[w])
        t2_vertical = _wmean(g["MAX_VERTICAL_LEAP"], g[w])
        # agilita'/sprint: piu' basso = piu' atletico -> invertiamo il segno a valle
        t2_agility = _wmean(g["LANE_AGILITY_TIME"], g[w])
        t2_sprint = _wmean(g["THREE_QUARTER_SPRINT"], g[w])
        imputed_share = float((min_share * g["is_imputed"].fillna(True).astype(int)).sum())

        # T3 esperienza playoff (somma roster, pesata per minuti)
        t3_depth_wsum = float((min_share * g["PLAYOFF_DEPTH_PRIOR"].fillna(0)).sum())
        t3_deep_runs = float((min_share * g["DEEP_RUNS_PRIOR"].fillna(0)).sum())
        t3_titles = float((min_share * g["TITLES_PRIOR"].fillna(0)).sum())

        # T4 tiro da 3 (volume + efficienza a livello squadra)
        fg3m = g["FG3M_TOT"].fillna(0).sum()
        fg3a = g["FG3A_TOT"].fillna(0).sum()
        t4_3pa_rate = float(fg3a / g.get("FGA_TOT", pd.Series([np.nan])).sum()) if "FGA_TOT" in g else np.nan
        t4_3p_pct = float(fg3m / fg3a) if fg3a > 0 else np.nan
        # numero di tiratori "affidabili" (>=100 tentativi, >=35%) -> spacing
        shooters = g[(g["FG3A_TOT"].fillna(0) >= 100) & (g["FG3_PCT"].fillna(0) >= 0.35)]
        t4_n_shooters = int(len(shooters))

        # T5 coinvolgimento non-superstar
        is_star = g["LEVEL"].isin(["superstar", "all_star"])
        t5_nonstar_min_share = float(min_share[~is_star].sum())
        t5_nonstar_pts_share = float(pts_share[~is_star].sum())
        t5_pts_entropy = _shannon_entropy(pts_share.to_numpy())
        t5_min_entropy = _shannon_entropy(min_share.to_numpy())

        # composizione roster (quanti per livello, pesato minuti)
        n_super = int((g["LEVEL"] == "superstar").sum())
        n_allstar = int((g["LEVEL"] == "all_star").sum())

        # --- FEATURE PER FASCIA DI RUOLO (richiesta utente) ---
        # fisico, tiro, esperienza ed efficacia SEPARATI per backcourt/wing/
        # frontcourt: cosi 'fisico sui lunghi ma piccoli solo tiratori' e' visibile
        # e i matchup per ruolo emergono.
        band_feats = _role_band_features(g, w)

        row = {
            "SEASON_START_YEAR": int(syear), "TEAM_ID": int(team_id),
            "TEAM_ABBREVIATION": g["TEAM_ABBREVIATION"].iloc[0],
            # T1
            "T1_hometown_minshare": t1_hometown_minshare,
            "T1_homegrown_core": t1_homegrown_core,
            # T2
            "T2_height": t2_height, "T2_wingspan": t2_wingspan, "T2_reach": t2_reach,
            "T2_vertical": t2_vertical, "T2_agility": t2_agility, "T2_sprint": t2_sprint,
            "T2_imputed_share": imputed_share,
            # T3
            "T3_playoff_depth": t3_depth_wsum, "T3_deep_runs": t3_deep_runs, "T3_titles": t3_titles,
            # T4
            "T4_3pa_rate": t4_3pa_rate, "T4_3p_pct": t4_3p_pct, "T4_n_shooters": t4_n_shooters,
            # T5
            "T5_nonstar_min_share": t5_nonstar_min_share,
            "T5_nonstar_pts_share": t5_nonstar_pts_share,
            "T5_pts_entropy": t5_pts_entropy, "T5_min_entropy": t5_min_entropy,
            # composizione
            "n_superstars": n_super, "n_allstars": n_allstar,
        }
        row.update(band_feats)
        rows.append(row)

    team_player = pd.DataFrame(rows)
    team_player = _attach_team_stats(team_player)
    team_player = _add_era_zscores(team_player)
    team_player.to_parquet(PROCESSED / "team_season_features.parquet", index=False)
    print(f"team_season_features: {len(team_player)} righe squadra-stagione, "
          f"{team_player['SEASON_START_YEAR'].nunique()} stagioni, "
          f"{team_player.shape[1]} colonne")
    return team_player


# feature dove il CONTESTO D'EPOCA conta (small-ball vs centri, boom del tiro):
# le normalizziamo rispetto alla media di stagione -> 'quanto sopra/sotto la
# norma di quell'anno'. Cosi essere tiratori/fisici pesa diversamente per era.
ERA_RELATIVE_COLS = [
    "T4_3pa_rate", "T4_3p_pct", "T4_n_shooters",
    "T2_height", "T2_wingspan", "PACE",
    "BAND_fron_minshare", "BAND_back_minshare",
    "BAND_fron_3p_pct", "BAND_back_height",
]


def _add_era_zscores(tp: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge z_<col> = (col - media_stagione) / std_stagione per le feature
    sensibili all'epoca. NON sostituisce le originali: le affianca."""
    out = tp.copy()
    for col in ERA_RELATIVE_COLS:
        if col not in out.columns:
            continue
        grp = out.groupby("SEASON_START_YEAR")[col]
        z = (out[col] - grp.transform("mean")) / grp.transform("std").replace(0, np.nan)
        out[f"zera_{col}"] = z.fillna(0.0)
    return out


def _attach_team_stats(tp: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge le team stats ufficiali (record, rating, pace, tiro da 3)."""
    frames = []
    for syear in sorted(tp["SEASON_START_YEAR"].unique()):
        season = f"{syear}-{str(syear+1)[-2:]}"
        adv = pd.read_parquet(RAW / "team_adv_rs" / f"{season}.parquet")
        base = pd.read_parquet(RAW / "team_base_rs" / f"{season}.parquet")
        m = adv[["TEAM_ID", "W", "L", "W_PCT", "OFF_RATING", "DEF_RATING",
                 "NET_RATING", "PACE", "TS_PCT", "EFG_PCT"]].copy()
        m = m.merge(base[["TEAM_ID", "FG3M", "FG3A", "FG3_PCT", "PTS"]], on="TEAM_ID", how="left")
        m["SEASON_START_YEAR"] = syear
        frames.append(m)
    team_stats = pd.concat(frames, ignore_index=True)
    tp = tp.merge(team_stats, on=["SEASON_START_YEAR", "TEAM_ID"], how="left")
    tp = _attach_home_away(tp)
    return tp


def _attach_home_away(tp: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge lo split casa/trasferta di Regular Season (fattore campo)."""
    ha_path = INTERIM / "home_away_rs.parquet"
    if not ha_path.exists():
        print("  ATTENZIONE: home_away_rs.parquet assente, salto lo split casa/trasferta")
        return tp
    ha = pd.read_parquet(ha_path)
    keep = ["SEASON_START_YEAR", "TEAM_ABBREVIATION",
            "HOME_WIN_PCT", "AWAY_WIN_PCT", "HOME_SPLIT", "HOME_NET", "AWAY_NET"]
    return tp.merge(ha[keep], on=["SEASON_START_YEAR", "TEAM_ABBREVIATION"], how="left")


if __name__ == "__main__":
    build_team_features()
