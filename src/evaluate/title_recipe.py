"""FASE V2-1b — La RICETTA DEL TITOLO: cosa porta lontano nei PLAYOFF?

recipe_search.py risponde a "cosa costruisce un buon net rating / record".
Ma la domanda VERA dell'utente e' "qual e' la ricetta per massimizzare le chance
di alzare il trofeo?". Il net rating e' solo un proxy: qui il target e' il
DEEP-RUN PESATO, cioe' quanto lontano arriva la squadra nei playoff.

Target ordinale DEEP_RUN (scelta utente):
  0 = fuori dai playoff, oppure eliminata al PRIMO TURNO
  1 = eliminata alle semifinali di conference  (ha vinto il round 1)
  2 = eliminata alle finali di conference       (ha vinto il round 2)
  3 = finalista NBA (persa la finale)            (ha vinto il round 3)
  4 = CAMPIONE NBA                               (ha vinto il round 4)

Derivato dai dati gia' presenti:
  - playoff_series.parquet : il WINNER di ogni serie e il ROUND
  - team_conference.parquet: l'universo completo delle squadre per stagione
                             (cosi le NON-playoff hanno deep-run = 0)

Metodo IDENTICO a recipe_search (forward-selection scorrelata, criterio R^2
OUT-OF-TIME, stessi input controllabili del roster) -> confrontabile direttamente:
si vede se "vincere il titolo" chiede ingredienti DIVERSI dal "dominare la RS".

Output: aggiunge la sezione 'DEEP_RUN' a data/processed/recipe.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluate.recipe_search import (CORR_THRESHOLD, OOT_END, OOT_START,
                                        _factor_summary, _report,
                                        _shap_direction, greedy_recipe,
                                        input_pool)

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

TARGET = "DEEP_RUN"


def build_deep_run() -> pd.DataFrame:
    """Una riga per (stagione, squadra) con il livello di deep-run [0..4].

    Una squadra che VINCE il round R e' arrivata almeno al livello R; il suo
    deep-run finale e' il round MASSIMO che ha vinto. Chi non vince nessuna serie
    (eliminata al primo turno o fuori dai playoff) = 0.
    """
    series = pd.read_parquet(INTERIM / "playoff_series.parquet")
    conf = pd.read_parquet(INTERIM / "team_conference.parquet")

    # round massimo VINTO da ciascuna squadra in ciascuna stagione
    won = series.groupby(["SEASON_START_YEAR", "WINNER"])["ROUND"].max()
    won = won.rename_axis(["SEASON_START_YEAR", "TEAM_ABBREVIATION"]).rename("DEEP_RUN")

    # universo completo: tutte le squadre per stagione (NON-playoff incluse -> 0)
    base = conf[["SEASON_START_YEAR", "TEAM_ABBREVIATION"]].drop_duplicates().copy()
    base = base.merge(won, on=["SEASON_START_YEAR", "TEAM_ABBREVIATION"], how="left")
    base["DEEP_RUN"] = base["DEEP_RUN"].fillna(0).astype(int)
    return base


def run() -> dict:
    tf = pd.read_parquet(PROCESSED / "team_season_features.parquet")
    deep = build_deep_run()
    df = tf.merge(deep, on=["SEASON_START_YEAR", "TEAM_ABBREVIATION"], how="inner")

    pool = input_pool(tf)
    df[pool] = df[pool].fillna(df[pool].median())

    print("\n" + "=" * 64)
    print(f"RICETTA DEL TITOLO — target = DEEP_RUN [0..4]  (pool={len(pool)} feature)")
    print("=" * 64)
    print("Distribuzione deep-run:",
          df["DEEP_RUN"].value_counts().sort_index().to_dict())

    rec = greedy_recipe(df, pool, TARGET)
    rec["direction"] = _shap_direction(df, rec["selected"], TARGET)
    rec["by_factor"] = _factor_summary(rec["direction"])
    rec["deep_run_distribution"] = {int(k): int(v) for k, v in
                                    df["DEEP_RUN"].value_counts().sort_index().items()}
    _report(rec)

    # merge nella ricetta esistente (net rating + record da recipe_search)
    recipe_path = PROCESSED / "recipe.json"
    full = json.loads(recipe_path.read_text()) if recipe_path.exists() else {}
    full[TARGET] = rec
    recipe_path.write_text(json.dumps(full, indent=2))
    print(f"\nAggiunta sezione DEEP_RUN a data/processed/recipe.json")

    _compare_recipes(full)
    return rec


def _compare_recipes(full: dict) -> None:
    """Confronto onesto: la ricetta del TITOLO chiede ingredienti diversi da
    quella del NET RATING?"""
    if "NET_RATING" not in full:
        return
    nr = set(full["NET_RATING"]["selected"])
    dr = set(full[TARGET]["selected"])
    print("\n" + "-" * 64)
    print("CONFRONTO: ricetta NET RATING vs ricetta TITOLO (deep-run)")
    print("-" * 64)
    print(f"  In COMUNE ({len(nr & dr)}): {sorted(nr & dr)}")
    print(f"  Solo per il TITOLO ({len(dr - nr)}): {sorted(dr - nr)}")
    print(f"  Solo per il NET RATING ({len(nr - dr)}): {sorted(nr - dr)}")


if __name__ == "__main__":
    run()
