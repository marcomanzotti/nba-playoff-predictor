"""FASE 5b — Cosa COSTRUISCE un buon record / net rating?

Il test della tesi (thesis_test) ha mostrato che a predire CHI VINCE LA SERIE
domina il record/net rating. Ma e' quasi una tautologia: vince chi e' piu' forte.
La vera domanda dell'utente: cosa RENDE forte una squadra? Cosa crea un buon
net rating?

Catena causale:
    feature giocatori (T1..T5)  ->  NET_RATING  ->  vittoria playoff
            INPUT                    MEDIATORE        OUTCOME
La Fase 5 ha testato l'ultima freccia. Qui testiamo la PRIMA: quanto le feature
derivate dai giocatori (fisico, esperienza, tiro, coinvolgimento, ruoli)
spiegano e costruiscono il net rating di squadra.

Modello: gradient boosting regressione, target = NET_RATING di squadra, input =
solo feature-tesi (NO record/rating, che sarebbero circolari). Validazione
temporale (alleno sul passato, testo sul futuro) per onesta'.

Output: data/processed/rating_drivers.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# input = feature derivate dai giocatori (la "materia prima" della squadra).
# Escludiamo record/rating ufficiali (sarebbero il target stesso) e gli z-epoca
# (ridondanti). Teniamo T1..T5 + per-ruolo + composizione roster.
INPUT_TAGS = ("T1_", "T2_", "T3_", "T4_", "T5_", "BAND_")
TARGET = "NET_RATING"


def _input_cols(tf: pd.DataFrame) -> list[str]:
    cols = [c for c in tf.columns
            if any(t in c for t in INPUT_TAGS) and not c.startswith("zera_")]
    cols += [c for c in ("n_superstars", "n_allstars") if c in tf.columns]
    return cols


def run() -> dict:
    tf = pd.read_parquet(PROCESSED / "team_season_features.parquet")
    cols = _input_cols(tf)
    data = tf.dropna(subset=[TARGET]).copy()
    data[cols] = data[cols].fillna(data[cols].median())

    # split temporale: train <=2018, test >=2019 (onesto, no leakage)
    train = data[data["SEASON_START_YEAR"] <= 2018]
    test = data[data["SEASON_START_YEAR"] >= 2019]

    model = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_lambda=2.0, random_state=42, n_jobs=4,
    )
    model.fit(train[cols], train[TARGET])

    r2_tr = r2_score(train[TARGET], model.predict(train[cols]))
    r2_te = r2_score(test[TARGET], model.predict(test[cols]))

    # SHAP: quali driver costruiscono il net rating, e in che direzione
    expl = shap.TreeExplainer(model)
    sv = pd.DataFrame(expl.shap_values(data[cols]), columns=cols)
    mean_abs = sv.abs().mean().sort_values(ascending=False)

    # direzione e contributo per fattore
    direction = {c: float(np.sign(np.corrcoef(data[c], sv[c])[0, 1]))
                 if data[c].std() > 0 else 0.0 for c in cols}
    by_factor = sv.abs().mean().groupby(_factor_of).sum().sort_values(ascending=False)

    results = {
        "target": TARGET,
        "n_inputs": len(cols),
        "r2_train": round(float(r2_tr), 3),
        "r2_test": round(float(r2_te), 3),
        "top_drivers": {c: round(float(v), 3) for c, v in mean_abs.head(15).items()},
        "direction": {c: direction[c] for c in mean_abs.head(15).index},
        "by_factor": {k: round(float(v), 3) for k, v in by_factor.items()},
    }
    (PROCESSED / "rating_drivers.json").write_text(json.dumps(results, indent=2))
    _report(results, mean_abs, direction)
    return results


def _factor_of(feat: str) -> str:
    for t in ("T1", "T2", "T3", "T4", "T5"):
        if f"{t}_" in feat:
            return t
    if feat.startswith("BAND_"):
        return "ROLE"
    if "star" in feat:
        return "ROSTER_QUALITY"
    return "OTHER"


def _report(r, mean_abs, direction) -> None:
    print("\n" + "=" * 64)
    print(f"COSA COSTRUISCE UN BUON NET RATING? (input = solo feature giocatori)")
    print("=" * 64)
    print(f"\nR2 train={r['r2_train']}  test={r['r2_test']}  "
          f"({r['n_inputs']} feature-giocatore spiegano il net rating)")
    print("\nContributo per FATTORE:")
    for k, v in r["by_factor"].items():
        print(f"  {k:16s} {v:.3f}")
    print("\nTop 12 DRIVER del net rating (con direzione):")
    for f, v in mean_abs.head(12).items():
        d = direction.get(f, 0)
        arrow = "↑ alza il rating" if d > 0 else ("↓ abbassa" if d < 0 else "·")
        print(f"  {f:28s} {v:.3f}  {arrow}")


if __name__ == "__main__":
    run()
