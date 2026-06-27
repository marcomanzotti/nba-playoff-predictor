"""Clustering #1 — archetipo di gioco + PROXY atletismo (tesi T2).

Problema: la Combine (fisico+atletismo) esiste solo dal 2000 e solo per i
giocatori che vi hanno partecipato. Per gli altri (pre-2000, undrafted,
internazionali) imputiamo valori fisici/atletici plausibili.

Strategia (spec §3.5):
  1. Consolidiamo la Combine (un valore per PLAYER_ID, il piu' completo).
  2. Costruiamo uno spazio di "stile di gioco + eta'" su cui TUTTI i
     giocatori-stagione hanno dati (da player_technical).
  3. Addestriamo un KNN sui giocatori CON Combine; per chi non l'ha,
     imputiamo le misure dai k vicini nello spazio stile+eta'.
  4. Flag is_imputed per tracciare l'incertezza.
  5. Assegniamo un archetipo (KMeans) sullo spazio fisico+stile risultante.

Tutto e' fittato sui dati storici e applicato per giocatore-stagione: poiche'
le misure fisiche sono ~stabili in carriera, usiamo un valore per giocatore
(non per stagione), mentre l'eta' modula le feature dipendenti dall'eta' a
valle (vedi aging curve, modulo separato).

Output (data/interim/):
  player_physical.parquet   PLAYER_ID + misure (reali o imputate) + is_imputed
  combine_consolidated.parquet
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

# Misure fisiche/atletiche che vogliamo (reali o imputate).
PHYS_COLS = [
    "HEIGHT_WO_SHOES", "WINGSPAN", "STANDING_REACH", "WEIGHT",
    "MAX_VERTICAL_LEAP", "STANDING_VERTICAL_LEAP",
    "LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT", "BODY_FAT_PCT",
]
# Feature di stile di gioco (per-game / advanced) usate per trovare i simili.
# Tutti i giocatori-stagione le hanno.
STYLE_COLS = [
    "PTS", "REB", "AST", "STL", "BLK", "FG3A", "FGA", "FTA",
    "AST_PCT", "REB_PCT", "USG_PCT", "TS_PCT", "FG3A_RATE",
]


def consolidate_combine() -> pd.DataFrame:
    """Un record per PLAYER_ID: media delle misure note (la Combine puo' avere
    piu' anni per lo stesso id in rari casi)."""
    frames = [pd.read_parquet(f) for f in glob.glob(str(RAW / "combine" / "*.parquet"))]
    c = pd.concat(frames, ignore_index=True)
    c = c[c["PLAYER_ID"] > 0].copy()
    keep = ["PLAYER_ID"] + [col for col in PHYS_COLS if col in c.columns]
    c = c[keep]
    # alcune misure arrivano come stringa/object (es. WEIGHT '184.60',
    # BODY_FAT_PCT) -> forziamo numerico prima di mediare.
    for col in [k for k in keep if k != "PLAYER_ID"]:
        c[col] = pd.to_numeric(c[col], errors="coerce")
    cons = c.groupby("PLAYER_ID").mean(numeric_only=True).reset_index()
    cons.to_parquet(INTERIM / "combine_consolidated.parquet", index=False)
    return cons


def _player_style_profile() -> pd.DataFrame:
    """Profilo di stile medio-carriera per giocatore (pesato sui minuti),
    piu' robusto di una singola stagione per trovare i simili."""
    tech = pd.read_parquet(INTERIM / "player_technical_rs.parquet")
    cols = [c for c in STYLE_COLS if c in tech.columns]
    # media pesata sui minuti totali per giocatore
    w = tech["MIN_TOT"].clip(lower=1)
    prof = (
        tech[cols].multiply(w, axis=0)
        .assign(PLAYER_ID=tech["PLAYER_ID"], _w=w)
        .groupby("PLAYER_ID")
        .apply(lambda g: pd.Series({c: g[c].sum() / g["_w"].sum() for c in cols}),
               include_groups=False)
        .reset_index()
    )
    return prof


def build_player_physical(k_neighbors: int = 7, n_archetypes: int = 8) -> pd.DataFrame:
    cons = consolidate_combine()
    style = _player_style_profile()
    phys_cols = [c for c in PHYS_COLS if c in cons.columns]

    # tutti i giocatori (anche senza combine) con un profilo di stile
    df = style.merge(cons, on="PLAYER_ID", how="left")
    df["is_imputed"] = df[phys_cols].isna().all(axis=1)

    style_cols = [c for c in STYLE_COLS if c in df.columns]

    # standardizziamo lo spazio di stile (imputando eventuali NaN di stile)
    style_imp = SimpleImputer(strategy="median").fit_transform(df[style_cols])
    style_scaled = StandardScaler().fit_transform(style_imp)

    have = ~df["is_imputed"].to_numpy()
    miss = df["is_imputed"].to_numpy()

    # imputazione KNN delle misure fisiche dai simili di stile
    if miss.sum() > 0 and have.sum() >= k_neighbors:
        nn = NearestNeighbors(n_neighbors=k_neighbors).fit(style_scaled[have])
        _, idx = nn.kneighbors(style_scaled[miss])
        donors = df.loc[have, phys_cols].to_numpy()
        for j, col in enumerate(phys_cols):
            imputed_vals = np.nanmean(donors[idx, j], axis=1)
            df.loc[miss, col] = imputed_vals

    # eventuali residui NaN (misure singole mancanti anche tra chi ha combine)
    df[phys_cols] = SimpleImputer(strategy="median").fit_transform(df[phys_cols])

    # archetipo di gioco: KMeans su stile + fisico standardizzati
    arch_feat = np.hstack([
        style_scaled,
        StandardScaler().fit_transform(df[phys_cols]),
    ])
    km = KMeans(n_clusters=n_archetypes, n_init=10, random_state=42)
    df["ARCHETYPE"] = km.fit_predict(arch_feat)

    out_cols = ["PLAYER_ID"] + phys_cols + ["is_imputed", "ARCHETYPE"]
    out = df[out_cols].copy()
    out.to_parquet(INTERIM / "player_physical.parquet", index=False)
    print(
        f"player_physical: {len(out)} giocatori | con Combine reale: {(~out['is_imputed']).sum()} "
        f"| imputati: {out['is_imputed'].sum()}"
    )
    print("Distribuzione archetipi:")
    print(out["ARCHETYPE"].value_counts().sort_index().to_string())
    return out


if __name__ == "__main__":
    build_player_physical()
