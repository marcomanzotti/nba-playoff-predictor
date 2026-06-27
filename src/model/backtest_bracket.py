"""Backtest del bracket (Fase 4).

Per ogni stagione del periodo di test, in modalita' walk-forward (allena solo
sul passato), simula il bracket col modello di serie e confronta col reale:
  - Monte Carlo  -> probabilita' di titolo/stadio per ogni squadra;
  - bracket modale (pick a prob>0.5) -> quante serie/round azzeccati;
  - quanto il modello ha dato al vero campione (prob assegnata) vs baseline.

Output (data/processed/):
  bracket_backtest_summary.json
  bracket_probabilities_{year}.parquet  (per le viz di Fase 6)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.bracket import (
    build_conference_bracket,
    most_likely_bracket,
    simulate_bracket,
)
from src.model.series_predictor import SeriesPredictor
from src.model.walkforward import _augment_symmetric, _feature_cols
from src.season_labels import label_with_title, season_label, title_year
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

# config vincente dal tuning su validation (Fase 3)
BEST_PARAMS = dict(
    objective="binary:logistic", eval_metric="logloss",
    subsample=0.8, colsample_bytree=0.7, random_state=42, n_jobs=4,
    n_estimators=150, max_depth=2, learning_rate=0.02,
    min_child_weight=5, reg_lambda=1.0,
)


def _fit_until(year: int, feat_cols: list[str]) -> XGBClassifier:
    """Allena il modello su tutte le serie PRIMA di `year` (walk-forward)."""
    df = pd.read_parquet(PROCESSED / "series_dataset.parquet")
    df = df.drop_duplicates(subset=["SEASON_START_YEAR", "ROUND", "TEAM1", "TEAM2"])
    df[feat_cols] = df[feat_cols].fillna(0.0)
    train = df[df["SEASON_START_YEAR"] < year]
    aug = _augment_symmetric(train, feat_cols)
    model = XGBClassifier(**BEST_PARAMS)
    model.fit(aug[feat_cols], aug["label"])
    return model


def _real_outcomes(year: int) -> dict:
    """Esiti reali: vincitori per round (per valutare i pick)."""
    s = pd.read_parquet(INTERIM / "playoff_series.parquet")
    sy = s[s["SEASON_START_YEAR"] == year]
    champ = sy.loc[sy["ROUND"] == 4, "WINNER"]
    return {
        "champion": champ.iloc[0] if len(champ) else None,
        "winners_by_round": {int(r): set(g["WINNER"]) for r, g in sy.groupby("ROUND")},
        "real_series": {(r["TEAM_A"], r["TEAM_B"]): r["WINNER"]
                        for _, r in sy.iterrows()},
    }


def backtest(years: range, n_sims: int = 5000) -> dict:
    feat_df = pd.read_parquet(PROCESSED / "series_dataset.parquet")
    feat_cols = _feature_cols(feat_df)

    results = {}
    for year in years:
        model = _fit_until(year, feat_cols)
        predictor = SeriesPredictor(model, feat_order=feat_cols)
        root = build_conference_bracket(year)

        # Monte Carlo
        probs = simulate_bracket(root, predictor, year, n_sims=n_sims)
        probs.to_parquet(PROCESSED / f"bracket_probabilities_{year}.parquet", index=False)

        # bracket modale + confronto con reale
        picks = most_likely_bracket(root, predictor, year)
        real = _real_outcomes(year)

        # accuratezza per-serie:
        #  - round 1: tutte le 8 coppie sono note -> sempre verificabili;
        #  - round>1: verificabili solo se l'accoppiamento e' avvenuto davvero
        #    (altrimenti il bracket ha gia' deviato: lo segnaliamo a parte).
        r1 = [p for p in picks if p["round"] == 1]
        r1_correct = sum(1 for p in r1 if _pick_correct(p, real))
        verifiable = [p for p in picks if _is_real_matchup(p, real)]
        verifiable_correct = sum(1 for p in verifiable if _pick_correct(p, real))

        champ_prob = float(probs.loc[probs["TEAM"] == real["champion"], "P_CHAMPION"].iloc[0]) \
            if real["champion"] in set(probs["TEAM"]) else 0.0
        results[int(year)] = {
            "season": season_label(year),
            "title_year": title_year(year),
            "champion_real": real["champion"],
            "champion_predicted_modal": _modal_champion(picks),
            "champion_hit_modal": _modal_champion(picks) == real["champion"],
            "round1_correct": r1_correct, "round1_total": len(r1),
            "verifiable_series_correct": verifiable_correct,
            "verifiable_series_total": len(verifiable),
            "prob_assigned_to_real_champion": round(champ_prob, 3),
            "top3_by_prob": probs.head(3)[["TEAM", "P_CHAMPION"]].to_dict("records"),
        }
        print(f"  {label_with_title(year)}: 1oturno {r1_correct}/{len(r1)} | "
              f"serie verificabili {verifiable_correct}/{len(verifiable)} | "
              f"campione {real['champion']} (p={champ_prob:.2f}) -> predetto "
              f"{results[int(year)]['champion_predicted_modal']}")

    (PROCESSED / "bracket_backtest_summary.json").write_text(json.dumps(results, indent=2, default=str))
    _summary(results, n_teams_uniform=16)
    return results


def _pick_correct(pick: dict, real: dict) -> bool:
    """Un pick e' corretto se la coppia (a,b) si e' davvero affrontata E il
    vincitore predetto coincide col reale. Per round >1 le coppie possono non
    combaciare (bracket diverge): allora il pick non si conta come verificabile,
    ma ai fini del backtest lo consideriamo errato se diverge dal reale."""
    key1 = (pick["team_a"], pick["team_b"])
    key2 = (pick["team_b"], pick["team_a"])
    real_series = real["real_series"]
    if key1 in real_series:
        return real_series[key1] == pick["pred_winner"]
    if key2 in real_series:
        return real_series[key2] == pick["pred_winner"]
    return False  # accoppiamento mai avvenuto -> il bracket ha gia' deviato


def _is_real_matchup(pick: dict, real: dict) -> bool:
    rs = real["real_series"]
    return (pick["team_a"], pick["team_b"]) in rs or (pick["team_b"], pick["team_a"]) in rs


def _modal_champion(picks: list[dict]) -> str:
    finals = [p for p in picks if p["round"] == 4]
    return finals[0]["pred_winner"] if finals else None


def _summary(results: dict, n_teams_uniform: int) -> None:
    yrs = list(results)
    r1_c = sum(r["round1_correct"] for r in results.values())
    r1_t = sum(r["round1_total"] for r in results.values())
    v_c = sum(r["verifiable_series_correct"] for r in results.values())
    v_t = sum(r["verifiable_series_total"] for r in results.values())
    champ_hits = sum(r["champion_hit_modal"] for r in results.values())
    mean_champ_prob = np.mean([r["prob_assigned_to_real_champion"] for r in results.values()])
    print("\n" + "=" * 60)
    print("BACKTEST BRACKET — RIEPILOGO")
    print("=" * 60)
    print(f"Stagioni: {yrs[0]}-{yrs[-1]}")
    print(f"1o turno azzeccato: {r1_c}/{r1_t} ({100*r1_c/r1_t:.1f}%)")
    print(f"Serie verificabili azzeccate: {v_c}/{v_t} ({100*v_c/v_t:.1f}%)")
    print(f"Campioni indovinati (bracket modale): {champ_hits}/{len(yrs)}")
    print(f"Prob media assegnata al vero campione: {mean_champ_prob:.3f} "
          f"(baseline uniforme 1/{n_teams_uniform} = {1/n_teams_uniform:.3f})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--sims", type=int, default=5000)
    args = ap.parse_args()
    backtest(range(args.start, args.end + 1), n_sims=args.sims)
