"""Modello di singola serie + validazione WALK-FORWARD 20/5/5 (Fase 3).

Schema temporale (orizzonte 1996-2025 = 30 stagioni):
  - TRAIN     : prime 20 stagioni (1996-2015)
  - VALIDATION: 2016-2020 (walk-forward espandente) -> qui si SCEGLIE la config
  - TEST      : 2021-2025 (intoccabile) -> si valuta UNA sola volta, alla fine

Il modello (XGBoost) usa TUTTE le feature differenziali disponibili (incl.
T1-T5: la tesi non limita le feature, e' solo lente di analisi a posteriori).

Anti-overfitting (decisione utente): teniamo tutte le feature ma TARIAMO gli
iperparametri (regolarizzazione) sul blocco VALIDATION via grid search,
scegliendo la config a logloss minimo. Il TEST non influenza nessuna scelta.

Simmetria: train augmentato con righe speculari; in predizione P(team1 vince)
si media vista diretta e speculare -> P(A>B) = 1 - P(B>A) per costruzione.
"""
from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

TRAIN_END = 2015
VAL_END = 2020
TEST_END = 2025

# Parametri fissi (conservativi) + griglia da tarare su validation.
BASE_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss",
    subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=4,
)
PARAM_GRID = {
    "n_estimators": [150, 300],
    "max_depth": [2, 3],
    "learning_rate": [0.02, 0.05],
    "min_child_weight": [5, 10],
    "reg_lambda": [1.0, 5.0],
}


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """Usa il set di feature SCORRELATO selezionato (select_features.py) se
    disponibile; altrimenti tutte le differenziali."""
    sel_path = PROCESSED / "selected_features.json"
    if sel_path.exists():
        kept = json.loads(sel_path.read_text())["kept"]
        return [c for c in kept if c in df.columns]
    extra = [c for c in ("H2H_DIFF", "HOME_COURT") if c in df.columns]
    return [c for c in df.columns if c.startswith("d_") or c.startswith("x_")] + extra


def _augment_symmetric(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    mirror = df.copy()
    mirror[feat_cols] = -df[feat_cols]
    mirror["label"] = 1 - df["label"]
    return pd.concat([df, mirror], ignore_index=True)


def _predict_symmetric(model, df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    p_direct = model.predict_proba(df[feat_cols])[:, 1]
    p_mirror = model.predict_proba(-df[feat_cols])[:, 1]
    return (p_direct + (1 - p_mirror)) / 2


def _metrics(y, p) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
    }


def _fit(train: pd.DataFrame, feat_cols: list[str], params: dict) -> XGBClassifier:
    aug = _augment_symmetric(train, feat_cols)
    model = XGBClassifier(**{**BASE_PARAMS, **params})
    model.fit(aug[feat_cols], aug["label"])
    return model


def _walkforward_preds(df, feat_cols, params, year_start, year_end) -> pd.DataFrame:
    """Walk-forward espandente: per ogni stagione in [start,end] allena su tutto
    il passato e predice quella stagione."""
    preds = []
    for year in range(year_start, year_end + 1):
        train = df[df["SEASON_START_YEAR"] < year]
        test = df[df["SEASON_START_YEAR"] == year]
        if len(test) == 0:
            continue
        model = _fit(train, feat_cols, params)
        p = _predict_symmetric(model, test, feat_cols)
        preds.append(pd.DataFrame({"year": year, "y": test["label"].to_numpy(), "p": p}))
    return pd.concat(preds, ignore_index=True)


def _grid_search_on_validation(df, feat_cols) -> tuple[dict, list]:
    """Sceglie la config a logloss minimo sul blocco VALIDATION (2016-2020).
    Il test NON viene mai toccato qui."""
    keys = list(PARAM_GRID)
    trials = []
    best = (None, np.inf)
    for combo in product(*[PARAM_GRID[k] for k in keys]):
        params = dict(zip(keys, combo))
        vp = _walkforward_preds(df, feat_cols, params, TRAIN_END + 1, VAL_END)
        m = _metrics(vp["y"], vp["p"])
        trials.append({"params": params, **m})
        if m["log_loss"] < best[1]:
            best = (params, m["log_loss"])
    trials.sort(key=lambda t: t["log_loss"])
    return best[0], trials


def _baselines(df: pd.DataFrame) -> dict:
    out = {}
    y = df["label"].to_numpy()
    for name, col in [("net_rating", "d_NET_RATING"), ("win_pct", "d_W_PCT"),
                      ("playoff_exp", "d_T3_playoff_depth")]:
        if col not in df.columns:
            continue
        z = df[col].fillna(0)
        z = (z - z.mean()) / (z.std() + 1e-9)
        p = 1 / (1 + np.exp(-z))
        out[name] = _metrics(y, p)
    return out


def run_walkforward() -> dict:
    df = pd.read_parquet(PROCESSED / "series_dataset.parquet")
    df = df.drop_duplicates(subset=["SEASON_START_YEAR", "ROUND", "TEAM1", "TEAM2"]).reset_index(drop=True)
    feat_cols = _feature_cols(df)
    df[feat_cols] = df[feat_cols].fillna(0.0)

    # 1) TUNING su validation (mai sul test)
    best_params, trials = _grid_search_on_validation(df, feat_cols)
    print("Config migliore (per logloss validation):")
    print("  ", best_params)

    # 2) metriche validation con la config scelta
    val_df = _walkforward_preds(df, feat_cols, best_params, TRAIN_END + 1, VAL_END)
    # 3) TEST finale (una sola volta) con la STESSA config
    test_df = _walkforward_preds(df, feat_cols, best_params, VAL_END + 1, TEST_END)

    eval_df = df[df["SEASON_START_YEAR"] > TRAIN_END]
    results = {
        "best_params": best_params,
        "top_trials": trials[:5],
        "val": {"model": _metrics(val_df["y"], val_df["p"]),
                "by_year": {int(yr): _metrics(g["y"], g["p"]) for yr, g in val_df.groupby("year")}},
        "test": {"model": _metrics(test_df["y"], test_df["p"]),
                 "by_year": {int(yr): _metrics(g["y"], g["p"]) for yr, g in test_df.groupby("year")}},
        "baselines": _baselines(eval_df),
        "feature_cols": feat_cols,
    }

    val_df.assign(split="val").to_parquet(PROCESSED / "walkforward_val_preds.parquet", index=False)
    test_df.assign(split="test").to_parquet(PROCESSED / "walkforward_test_preds.parquet", index=False)
    (PROCESSED / "walkforward_results.json").write_text(json.dumps(results, indent=2))

    _print_report(results)
    return results


def _print_report(r: dict) -> None:
    print("\n" + "=" * 60)
    print("WALK-FORWARD 20/5/5 — RISULTATI (config tarata su validation)")
    print("=" * 60)
    print("\n[VALIDATION 2016-2020]")
    _pm(r["val"]["model"])
    print("\n[TEST 2021-2025]  (verdetto finale, config non toccata dal test)")
    _pm(r["test"]["model"])
    print("\n[BASELINE su val+test]")
    for name, m in r["baselines"].items():
        print(f"  {name:14s}: acc={m['accuracy']:.3f}  logloss={m['log_loss']:.3f}  brier={m['brier']:.3f}")
    print("\n[TEST per stagione]")
    for yr, m in r["test"]["by_year"].items():
        print(f"  {yr}: acc={m['accuracy']:.3f} ({m['n']} serie)  logloss={m['log_loss']:.3f}")


def _pm(m: dict) -> None:
    print(f"  n={m['n']}  accuracy={m['accuracy']:.3f}  logloss={m['log_loss']:.3f}  brier={m['brier']:.3f}")


if __name__ == "__main__":
    run_walkforward()
