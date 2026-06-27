"""Dataset a livello SERIE per il modello di singola serie (Fase 3).

Una riga per serie playoff storica: feature = DIFFERENZA tra le due squadre
(team1 - team2) per ogni metrica disponibile, piu' il contesto (round,
vantaggio campo via record, head-to-head di RS). Target = team1 vince la serie.

IMPORTANTE (decisione utente): il modello usa TUTTE le feature disponibili,
incluse le T1-T5. La tesi NON limita le feature: T1-T5 serviranno solo come
LENTE di analisi a posteriori (Fase 5, SHAP/ablation).

Simmetria: per non far imparare un bias di posizione, ogni serie genera DUE
righe speculari (team1-team2, label=1/0) e (team2-team1, label opposto). Cosi
per costruzione il modello tende a P(A>B) = 1 - P(B>A).

Output (data/processed/):
  series_dataset.parquet   2 righe per serie (augmented), con META + diff feature
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

# Feature squadra da differenziare (tutte le numeriche tranne chiavi).
_DROP_FROM_FEATURES = {"SEASON_START_YEAR", "TEAM_ID"}


def _team_feature_cols(tf: pd.DataFrame) -> list[str]:
    num = tf.select_dtypes("number").columns.tolist()
    return [c for c in num if c not in _DROP_FROM_FEATURES]


def build_series_dataset() -> pd.DataFrame:
    tf = pd.read_parquet(PROCESSED / "team_season_features.parquet")
    series = pd.read_parquet(INTERIM / "playoff_series.parquet")
    h2h = pd.read_parquet(INTERIM / "h2h_records.parquet")
    conf = pd.read_parquet(INTERIM / "team_conference.parquet")

    feat_cols = _team_feature_cols(tf)
    tf_idx = tf.set_index(["SEASON_START_YEAR", "TEAM_ABBREVIATION"])

    # head-to-head RS: vittorie di TEAM su OPP in quella stagione
    h2h_idx = h2h.set_index(["SEASON_START_YEAR", "TEAM", "OPP"])
    # seed ufficiale per il vantaggio campo della serie (chi ha seed migliore
    # = numero piu' basso gioca in casa gara 1,2,5,7)
    seed_idx = conf.set_index(["SEASON_START_YEAR", "TEAM_ABBREVIATION"])["SEED"]

    rows = []
    skipped = 0
    for _, s in series.iterrows():
        syear = int(s["SEASON_START_YEAR"])
        a, b = s["TEAM_A"], s["TEAM_B"]
        key_a, key_b = (syear, a), (syear, b)
        if key_a not in tf_idx.index or key_b not in tf_idx.index:
            skipped += 1
            continue
        fa = tf_idx.loc[key_a, feat_cols]
        fb = tf_idx.loc[key_b, feat_cols]
        # se per qualche motivo ci sono duplicati, prendi la prima riga
        if isinstance(fa, pd.DataFrame):
            fa = fa.iloc[0]
        if isinstance(fb, pd.DataFrame):
            fb = fb.iloc[0]

        # head-to-head RS (A vs B): differenza vittorie nel confronto diretto
        h2h_a = _h2h_wins(h2h_idx, syear, a, b)
        h2h_b = _h2h_wins(h2h_idx, syear, b, a)
        h2h_diff = h2h_a - h2h_b

        # vantaggio campo della serie: chi ha seed migliore (numero piu' basso)
        # gioca in casa. HOME_COURT = +1 se A ha il fattore campo, -1 se B.
        home_court = _home_court(seed_idx, syear, a, b)

        diff = (fa - fb)
        meta = {
            "SEASON_START_YEAR": syear, "ROUND": int(s["ROUND"]),
            "TEAM1": a, "TEAM2": b,
            "label": int(s["WINNER"] == a),
        }
        row = {**meta, "H2H_DIFF": h2h_diff, "HOME_COURT": home_court}
        for c in feat_cols:
            row[f"d_{c}"] = float(diff[c]) if pd.notna(diff[c]) else np.nan
        rows.append(row)

    base = pd.DataFrame(rows)

    # --- data augmentation speculare ---
    # tutte le feature differenziali (incl. H2H_DIFF e HOME_COURT) sono
    # ANTISIMMETRICHE: invertendo le squadre cambiano segno.
    mirror = base.copy()
    mirror["TEAM1"], mirror["TEAM2"] = base["TEAM2"], base["TEAM1"]
    mirror["label"] = 1 - base["label"]
    mirror["H2H_DIFF"] = -base["H2H_DIFF"]
    mirror["HOME_COURT"] = -base["HOME_COURT"]
    diff_cols = [c for c in base.columns if c.startswith("d_")]
    mirror[diff_cols] = -base[diff_cols]

    full = pd.concat([base, mirror], ignore_index=True)
    full.to_parquet(PROCESSED / "series_dataset.parquet", index=False)
    print(
        f"series_dataset: {len(full)} righe ({len(base)} serie x2 augmented), "
        f"saltate {skipped} serie senza feature, "
        f"{len([c for c in full.columns if c.startswith('d_')]) + 1} feature differenziali"
    )
    print(f"  label balance: {full['label'].mean():.3f} (atteso ~0.5 per simmetria)")
    return full


def _h2h_wins(h2h_idx: pd.DataFrame, syear: int, team: str, opp: str) -> float:
    try:
        return float(h2h_idx.loc[(syear, team, opp), "WINS"])
    except KeyError:
        return 0.0


def _home_court(seed_idx: pd.Series, syear: int, a: str, b: str) -> float:
    """+1 se A ha il vantaggio campo (seed migliore = numero piu' basso),
    -1 se ce l'ha B, 0 se ignoto/pari."""
    try:
        sa = float(seed_idx.loc[(syear, a)])
        sb = float(seed_idx.loc[(syear, b)])
    except KeyError:
        return 0.0
    if sa < sb:
        return 1.0
    if sb < sa:
        return -1.0
    return 0.0


if __name__ == "__main__":
    build_series_dataset()
