"""Selezione di un set di feature SCORRELATO (richiesta utente).

XGBoost coglie da solo interazioni e relazioni non-lineari: feature ridondanti
NON aiutano e anzi DILUISCONO la feature importance (il merito si spalma tra
colonne gemelle), rendendo difficile leggere cosa conta davvero -> fatale per
il test della tesi (Fase 5).

Strategia: raggruppiamo le feature per |correlazione| > soglia e teniamo UN
rappresentante per gruppo, scelto con una priorita' che privilegia:
  1) le feature-tesi T1..T5 (vogliamo poterle leggere in chiaro);
  2) le piu' interpretabili / informative;
scartando i duplicati tecnici (W vs L vs W_PCT, zera_* vs originale, ecc.).

Salviamo l'elenco delle feature tenute in data/processed/selected_features.json,
usato dal modello.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

CORR_THRESHOLD = 0.85

# Rimozioni esplicite (duplicati ovvi): teniamo la prima, scartiamo le altre.
# La selezione automatica gestisce il resto.
HARD_DROP_PREFIXES = ("d_zera_",)  # gli z-epoca duplicano le originali (r~0.99)


def _priority(feat: str) -> int:
    """Priorita' di MANTENIMENTO (piu' alto = preferito come rappresentante)."""
    # le feature-tesi T1..T5 hanno priorita' massima (leggibilita' tesi)
    if any(t in feat for t in ("T1_", "T2_", "T3_", "T4_", "T5_")):
        return 5
    # matchup espliciti + interazioni di stile (il punto dell'esercizio V2)
    if feat in ("H2H_DIFF", "HOME_COURT") or feat.startswith("x_"):
        return 4
    # rating sintetici preferiti ai grezzi
    if "NET_RATING" in feat or feat == "d_W_PCT":
        return 3
    # feature per fascia di ruolo (granularita' richiesta)
    if feat.startswith("d_BAND_"):
        return 2
    return 1


def select_features(corr_threshold: float = CORR_THRESHOLD) -> list[str]:
    sd = pd.read_parquet(PROCESSED / "series_dataset.parquet")
    all_feats = [c for c in sd.columns
                 if c.startswith("d_") or c.startswith("x_")
                 or c in ("H2H_DIFF", "HOME_COURT")]
    # drop esplicito degli z-epoca ridondanti
    feats = [f for f in all_feats if not f.startswith(HARD_DROP_PREFIXES)]

    X = sd[feats].fillna(0)
    corr = X.corr().abs()

    # greedy: ordiniamo per priorita' (poi varianza); aggiungiamo una feature
    # solo se non e' troppo correlata con quelle gia' tenute.
    order = sorted(feats, key=lambda f: (_priority(f), X[f].var()), reverse=True)
    kept: list[str] = []
    for f in order:
        if all(corr.loc[f, k] <= corr_threshold for k in kept):
            kept.append(f)

    # ordine stabile (alfabetico) per riproducibilita'
    kept = sorted(kept)
    dropped = sorted(set(all_feats) - set(kept))
    out = {"kept": kept, "dropped": dropped,
           "n_kept": len(kept), "n_dropped": len(dropped),
           "corr_threshold": corr_threshold}
    (PROCESSED / "selected_features.json").write_text(json.dumps(out, indent=2))
    print(f"Feature selezionate: {len(kept)} tenute, {len(dropped)} scartate "
          f"(su {len(all_feats)} totali)")
    print("\nTenute (per categoria):")
    for cat, pref in [("Tesi T1-T5", lambda f: any(t in f for t in ('T1_','T2_','T3_','T4_','T5_'))),
                      ("Matchup", lambda f: f in ('H2H_DIFF','HOME_COURT')),
                      ("Per ruolo", lambda f: f.startswith('d_BAND_')),
                      ("Squadra", lambda f: True)]:
        sel = [f for f in kept if pref(f)]
        if sel:
            print(f"  {cat}: {len(sel)}")
    return kept


if __name__ == "__main__":
    select_features()
