"""FASE 5 — Test della TESI (cuore del progetto).

Domanda: 'cosa fa vincere in NBA?'. La tesi dell'utente e' che contino
T1 hometown, T2 fisico/atletismo, T3 esperienza playoff, T4 tiro da 3,
T5 coinvolgimento dei non-superstar (+ granularita' per ruolo).

Misuriamo il loro valore in 3 modi RIGOROSI (walk-forward 20/5/5, mai sul test
per scegliere), confrontando set di feature diversi:

  A) BASELINE      : solo record/rating/fattore campo/matchup (NO tesi)
  B) COMPLETO      : baseline + tutte le feature-tesi
  C) SOLO-TESI     : solo le feature-tesi (NO record/rating)

Se la tesi 'spiega cosa fa vincere':
  - B deve battere A (le feature-tesi aggiungono valore predittivo);
  - C da sola deve predire meglio del caso (la tesi ha potere autonomo).

Output: data/processed/thesis_test_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from src.model.walkforward import (
    BASE_PARAMS,
    TRAIN_END,
    VAL_END,
    TEST_END,
    _augment_symmetric,
    _feature_cols,
    _predict_symmetric,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# config vincente (tuning su validation, Fase 3/ripulitura)
FIXED = dict(n_estimators=150, max_depth=3, learning_rate=0.02,
             min_child_weight=5, reg_lambda=5.0)

THESIS_TAGS = ("T1_", "T2_", "T3_", "T4_", "T5_", "BAND_")


def _split_feature_sets(all_feats: list[str]) -> dict[str, list[str]]:
    thesis = [f for f in all_feats if any(t in f for t in THESIS_TAGS)]
    baseline = [f for f in all_feats if f not in thesis]
    return {
        "baseline": baseline,         # A: no tesi
        "complete": all_feats,        # B: tutto
        "thesis_only": thesis,        # C: solo tesi
    }


def _metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, (p >= 0.5).astype(int))), 4),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
    }


def _walkforward(df, feat_cols, y0, y1) -> pd.DataFrame:
    preds = []
    for year in range(y0, y1 + 1):
        train = df[df["SEASON_START_YEAR"] < year]
        test = df[df["SEASON_START_YEAR"] == year]
        if len(test) == 0:
            continue
        aug = _augment_symmetric(train, feat_cols)
        model = XGBClassifier(**{**BASE_PARAMS, **FIXED})
        model.fit(aug[feat_cols], aug["label"])
        p = _predict_symmetric(model, test, feat_cols)
        preds.append(pd.DataFrame({"year": year, "y": test["label"].to_numpy(), "p": p}))
    return pd.concat(preds, ignore_index=True)


def run_thesis_test() -> dict:
    df = pd.read_parquet(PROCESSED / "series_dataset.parquet")
    df = df.drop_duplicates(subset=["SEASON_START_YEAR", "ROUND", "TEAM1", "TEAM2"]).reset_index(drop=True)
    all_feats = _feature_cols(df)
    df[all_feats] = df[all_feats].fillna(0.0)

    sets = _split_feature_sets(all_feats)
    results = {"feature_sets": {k: len(v) for k, v in sets.items()},
               "thesis_features": sets["thesis_only"],
               "baseline_features": sets["baseline"]}

    for name, cols in sets.items():
        val = _walkforward(df, cols, TRAIN_END + 1, VAL_END)
        test = _walkforward(df, cols, VAL_END + 1, TEST_END)
        results[name] = {
            "n_features": len(cols),
            "validation": _metrics(val["y"], val["p"]),
            "test": _metrics(test["y"], test["p"]),
        }

    (PROCESSED / "thesis_test_results.json").write_text(json.dumps(results, indent=2))
    _report(results)
    return results


def _report(r: dict) -> None:
    print("\n" + "=" * 66)
    print("FASE 5 — TEST DELLA TESI (ablation walk-forward)")
    print("=" * 66)
    print(f"\nFeature: baseline={r['feature_sets']['baseline']} | "
          f"tesi={r['feature_sets']['thesis_only']} | completo={r['feature_sets']['complete']}")
    print(f"\n{'MODELLO':<16} {'VAL acc':>8} {'VAL ll':>8} {'TEST acc':>9} {'TEST ll':>8} {'TEST brier':>11}")
    print("-" * 66)
    labels = {"baseline": "A) Baseline", "complete": "B) Completo", "thesis_only": "C) Solo-tesi"}
    for key in ("baseline", "complete", "thesis_only"):
        m = r[key]
        v, t = m["validation"], m["test"]
        print(f"{labels[key]:<16} {v['accuracy']:>8.3f} {v['log_loss']:>8.3f} "
              f"{t['accuracy']:>9.3f} {t['log_loss']:>8.3f} {t['brier']:>11.3f}")
    print("-" * 66)
    # interpretazione automatica
    a, b, c = r["baseline"]["test"], r["complete"]["test"], r["thesis_only"]["test"]
    print("\nLETTURA:")
    d_ll = a["log_loss"] - b["log_loss"]
    print(f"  Completo vs Baseline (logloss): {d_ll:+.3f} "
          f"({'la tesi AGGIUNGE valore' if d_ll > 0 else 'la tesi NON aggiunge sul test'})")
    print(f"  Solo-tesi test accuracy: {c['accuracy']:.3f} "
          f"({'> caso (0.5): la tesi predice da sola' if c['accuracy'] > 0.5 else '~caso'})")


if __name__ == "__main__":
    run_thesis_test()
