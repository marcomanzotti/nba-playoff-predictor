"""FASE V2-1c — La RICETTA NON-TAUTOLOGICA: cosa fa vincere A PARITA' DI TALENTO.

CRITICA dell'utente (giusta): "la qualita' dei giocatori non vuol dire nulla;
ovvio che giocatori forti aiutano a vincere. Se la conclusione e' 'serve vincere
tanto e avere giocatori forti', e' inutile, ci arrivavo senza statistica."

RISPOSTA: separiamo TALENTO (esito) da STRUTTURA (leve costruibili) e cerchiamo la
ricetta in DUE modi complementari:

  RICETTA A — "solo struttura":
      target = DEEP_RUN (quanto lontano arriva nei playoff)
      input  = SOLO feature strutturali (fisico, tiro, esperienza, composizione
               per ruolo, pace). NIENTE livello/impatto dei giocatori.
      -> quali caratteristiche costruibili predicono il deep-run, ignorando del
         tutto il talento. R^2 piu' basso (onesto), ma azionabile.

  RICETTA B — "a parita' di talento" (OVER-PERFORMANCE):
      1) modello-TALENTO: predice DEEP_RUN usando SOLO le feature di talento
         (livello per ruolo, n. superstar...). E' il "deep-run atteso dal talento".
      2) OVER_PERF = DEEP_RUN_reale - DEEP_RUN_atteso_dal_talento  (il residuo).
         > 0 = la squadra va OLTRE quanto il suo talento prevede; < 0 = sotto.
      3) cerchiamo quali feature STRUTTURALI spiegano OVER_PERF.
      -> QUI c'e' la vera ricetta: dati due roster ugualmente forti, cosa fa la
         differenza? (tiro? esperienza? dimensione? diffusione?)

Tutto out-of-time (walk-forward), riusando il motore di recipe_search.

Output: data/processed/recipe_structural.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate.recipe_search import (XGB_PARAMS, _factor_summary, _report,
                                        _shap_direction, greedy_recipe,
                                        input_pool)
from src.evaluate.title_recipe import build_deep_run

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

OOT_START, OOT_END = 2016, 2025


def _talent_expected_deeprun(df: pd.DataFrame, talent_cols: list[str]) -> np.ndarray:
    """Deep-run ATTESO solo dal talento, stimato OUT-OF-TIME (walk-forward) cosi
    il residuo non e' inquinato da leakage. Per le stagioni troppo presto per il
    walk-forward usiamo un modello full come fallback (servono solo da controllo)."""
    from xgboost import XGBRegressor

    pred = pd.Series(np.nan, index=df.index)
    for year in range(OOT_START, OOT_END + 1):
        tr = df[df["SEASON_START_YEAR"] < year]
        te = df[df["SEASON_START_YEAR"] == year]
        if len(te) == 0 or len(tr) < 50:
            continue
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(tr[talent_cols], tr["DEEP_RUN"])
        pred.loc[te.index] = m.predict(te[talent_cols])
    # fallback per le stagioni iniziali (fuori dalla finestra OOT): modello full
    mask = pred.isna()
    if mask.any():
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(df[talent_cols], df["DEEP_RUN"])
        pred.loc[mask] = m.predict(df.loc[mask, talent_cols])
    return pred.to_numpy()


def run() -> dict:
    tf = pd.read_parquet(PROCESSED / "team_season_features.parquet")
    deep = build_deep_run()
    df = tf.merge(deep, on=["SEASON_START_YEAR", "TEAM_ABBREVIATION"], how="inner").reset_index(drop=True)

    struct_cols = input_pool(tf, "structure")
    talent_cols = input_pool(tf, "talent")
    df[struct_cols] = df[struct_cols].fillna(df[struct_cols].median())
    df[talent_cols] = df[talent_cols].fillna(df[talent_cols].median())

    print(f"\nSTRUTTURA: {len(struct_cols)} feature  |  TALENTO (controllo): {len(talent_cols)} feature")

    results = {"structure_pool": len(struct_cols), "talent_pool": len(talent_cols),
               "oot_window": [OOT_START, OOT_END]}

    # ---------- RICETTA A: solo struttura -> DEEP_RUN ----------
    print("\n" + "=" * 70)
    print("RICETTA A — SOLO STRUTTURA che predice il DEEP-RUN (niente talento)")
    print("=" * 70)
    recA = greedy_recipe(df, struct_cols, "DEEP_RUN")
    recA["direction"] = _shap_direction(df, recA["selected"], "DEEP_RUN")
    recA["by_factor"] = _factor_summary(recA["direction"])
    _report(recA)
    results["structure_only"] = recA

    # ---------- RICETTA B: over-performance a parita' di talento ----------
    df["DEEP_RUN_EXPECTED"] = _talent_expected_deeprun(df, talent_cols)
    df["OVER_PERF"] = df["DEEP_RUN"] - df["DEEP_RUN_EXPECTED"]
    print("\n" + "=" * 70)
    print("RICETTA B — OVER-PERFORMANCE: cosa fa la differenza A PARITA' DI TALENTO")
    print("=" * 70)
    print(f"OVER_PERF: media={df['OVER_PERF'].mean():.3f} (atteso ~0), "
          f"std={df['OVER_PERF'].std():.3f}")
    recB = greedy_recipe(df, struct_cols, "OVER_PERF")
    recB["direction"] = _shap_direction(df, recB["selected"], "OVER_PERF")
    recB["by_factor"] = _factor_summary(recB["direction"])
    _report(recB)
    results["over_performance"] = recB

    (PROCESSED / "recipe_structural.json").write_text(json.dumps(results, indent=2))
    print("\nSalvato data/processed/recipe_structural.json")
    _verdict(recA, recB)
    return results


def _verdict(recA: dict, recB: dict) -> None:
    print("\n" + "=" * 70)
    print("VERDETTO — la ricetta NON-tautologica")
    print("=" * 70)
    print(f"\nA) Sola STRUTTURA -> deep-run:  R2_oot={recA['r2_oot_final']} "
          f"({recA['n_selected']} leve)")
    print(f"B) A PARITA' di talento (over-perf):  R2_oot={recB['r2_oot_final']} "
          f"({recB['n_selected']} leve)")
    print("\nLe leve STRUTTURALI che contano di piu' a parita' di talento:")
    for f in recB["selected"][:6]:
        d = recB["direction"][f]
        arrow = "↑" if d["direction"] > 0 else "↓"
        print(f"   {arrow} {f}")


if __name__ == "__main__":
    run()
