"""FASE 5 — Analisi SHAP del modello completo.

SHAP scompone ogni predizione nei contributi delle singole feature: ci dice non
solo QUANTO conta una feature, ma in CHE DIREZIONE (piu' esperienza playoff ->
piu' probabilita' di vincere?) e con che intensita'.

Aggreghiamo i valori SHAP per:
  - feature singola (importanza media |SHAP|);
  - FATTORE-TESI (T1..T5): sommiamo i contributi di tutte le feature di ogni
    fattore -> 'quanto pesa, nel complesso, il tiro da 3 (T4)?'.

Salva: data/processed/shap_values.parquet (per le viz Fase 6)
       data/processed/shap_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from src.model.walkforward import (
    BASE_PARAMS,
    VAL_END,
    _augment_symmetric,
    _feature_cols,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

FIXED = dict(n_estimators=150, max_depth=3, learning_rate=0.02,
             min_child_weight=5, reg_lambda=5.0)


def _thesis_factor(feat: str) -> str:
    for t in ("T1", "T2", "T3", "T4", "T5"):
        if f"{t}_" in feat:
            return t
    if feat.startswith("d_BAND_"):
        return "ROLE"        # feature per fascia di ruolo
    if feat in ("HOME_COURT", "d_HOME_WIN_PCT", "d_AWAY_WIN_PCT", "d_HOME_SPLIT"):
        return "HOME_COURT"
    if "RATING" in feat or feat in ("d_PTS", "d_EFG_PCT", "d_PACE", "d_FG3A"):
        return "TEAM_STATS"
    if feat == "H2H_DIFF":
        return "MATCHUP_H2H"
    if "star" in feat:
        return "ROSTER_QUALITY"
    return "OTHER"


def run_shap() -> dict:
    df = pd.read_parquet(PROCESSED / "series_dataset.parquet")
    df = df.drop_duplicates(subset=["SEASON_START_YEAR", "ROUND", "TEAM1", "TEAM2"]).reset_index(drop=True)
    feats = _feature_cols(df)
    df[feats] = df[feats].fillna(0.0)

    # alleniamo sul training+validation (fino a VAL_END) e spieghiamo TUTTE le
    # serie del periodo (per leggere la struttura, non per validare).
    train = df[df["SEASON_START_YEAR"] <= VAL_END]
    aug = _augment_symmetric(train, feats)
    model = XGBClassifier(**{**BASE_PARAMS, **FIXED})
    model.fit(aug[feats], aug["label"])

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(df[feats])
    shap_df = pd.DataFrame(sv, columns=feats)
    shap_df.to_parquet(PROCESSED / "shap_values.parquet", index=False)

    # importanza media |SHAP| per feature
    mean_abs = shap_df.abs().mean().sort_values(ascending=False)

    # contributo aggregato per FATTORE-TESI
    factor_map = {f: _thesis_factor(f) for f in feats}
    by_factor = (
        shap_df.abs().mean()
        .groupby(factor_map).sum()
        .sort_values(ascending=False)
    )

    # direzione: correlazione tra valore feature e suo SHAP (segno dell'effetto)
    direction = {}
    for f in feats:
        if df[f].std() > 0:
            direction[f] = float(np.sign(np.corrcoef(df[f], shap_df[f])[0, 1]))
        else:
            direction[f] = 0.0

    summary = {
        "top_features": {f: round(float(v), 4) for f, v in mean_abs.head(20).items()},
        "by_factor": {k: round(float(v), 4) for k, v in by_factor.items()},
        "direction": direction,
    }
    (PROCESSED / "shap_summary.json").write_text(json.dumps(summary, indent=2))
    _report(mean_abs, by_factor, direction)
    return summary


def _report(mean_abs, by_factor, direction) -> None:
    print("\n" + "=" * 60)
    print("FASE 5 — ANALISI SHAP (modello completo)")
    print("=" * 60)
    print("\nContributo per FATTORE (somma |SHAP| delle sue feature):")
    for k, v in by_factor.items():
        print(f"  {k:16s} {v:.4f}")
    print("\nTop 12 feature singole (|SHAP| medio, con direzione):")
    for f, v in mean_abs.head(12).items():
        d = direction.get(f, 0)
        arrow = "↑ favorisce" if d > 0 else ("↓ sfavorisce" if d < 0 else "·")
        print(f"  {f:30s} {v:.4f}  {arrow}")


if __name__ == "__main__":
    run_shap()
