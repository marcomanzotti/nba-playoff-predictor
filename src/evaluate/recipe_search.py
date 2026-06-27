"""FASE V2-1 — La RICETTA: quali feature COSTRUISCONO un buon net rating / record?

Differenza chiave rispetto a what_builds_rating.py (V1):
  - V1 partiva SOLO dalle 5 ipotesi T1..T5 dell'utente e usava uno split fisso.
  - QUI partiamo da TUTTE le feature-giocatore disponibili (la "materia prima"
    controllabile nella costruzione del roster) e lasciamo che i DATI scelgano il
    sottoinsieme SCORRELATO di X feature che massimizza la spiegazione.

Domanda dell'utente (testuale): "voglio prendere tutte le feature disponibili e
capire quali delle X spiegano al meglio il net rating, finche non troviamo la
combinazione di feature (preferibilmente scorrelate) che massimizza la
spiegazione del net rating E del record".

METODO (onesto, anti-overfitting):
  - La selezione e' GREEDY FORWARD: a ogni passo aggiunge la feature che alza di
    piu' l'R^2, ma SOLO se non e' troppo correlata (|corr| <= 0.85, stessa soglia
    di select_features.py) con quelle gia' scelte.
  - Il criterio NON e' l'R^2 in-sample (con 46 feature su 892 righe troverebbe
    sempre R^2 alto = overfitting), ma l'R^2 OUT-OF-TIME: alleni sul passato,
    misuri sul futuro (walk-forward, come walkforward.py::_walkforward_preds).
  - Ci si ferma quando l'R^2 out-of-time smette di salire (early stop): cosi
    emerge "quante feature BASTANO".

Le feature-OUTCOME (NET_RATING, W_PCT, OFF/DEF_RATING, TS_PCT, EFG_PCT, PTS,
split casa/trasferta) sono il target o suoi sinonimi -> ESCLUSE dagli input: la
ricetta deve usare solo ingredienti controllabili nel roster.

Output: data/processed/recipe.json  (sezioni 'NET_RATING' e 'W_PCT')
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

# --- target / sinonimi del target: NON sono ingredienti, sono esiti -> esclusi.
OUTCOME_COLS = {
    "NET_RATING", "OFF_RATING", "DEF_RATING", "W", "L", "W_PCT",
    "TS_PCT", "EFG_PCT", "PTS", "FG3M", "FG3A", "FG3_PCT",
    "HOME_WIN_PCT", "AWAY_WIN_PCT", "HOME_SPLIT", "HOME_NET", "AWAY_NET",
}
KEY_COLS = {"SEASON_START_YEAR", "TEAM_ID", "TEAM_ABBREVIATION"}

# --- TALENTO vs STRUTTURA (decisione utente, anti-tautologia) ---------------
# Il TALENTO (livello/impatto/produzione del giocatore) e' un ESITO: dire "per
# vincere servono giocatori forti" e' una tautologia, non una ricetta. Lo
# separiamo per poterlo NEUTRALIZZARE (controllarlo) e cercare invece le leve
# STRUTTURALI (caratteristiche costruibili e oggettive: fisico, tiro, esperienza,
# composizione per ruolo, pace) che fanno la differenza A PARITA' di talento.
TALENT_KEYWORDS = ("_level", "n_superstars", "n_allstars",
                   "nonstar_pts", "nonstar_min", "pts_entropy", "min_entropy")


def is_talent(feat: str) -> bool:
    return any(k in feat for k in TALENT_KEYWORDS)


# soglia di scorrelazione (coerente con select_features.py)
CORR_THRESHOLD = 0.85
# walk-forward out-of-time: per ogni anno in questo range alleniamo sul passato
# e misuriamo l'R^2 su quell'anno (mai visto). Media su piu' cut = stima onesta.
OOT_START, OOT_END = 2016, 2025
MIN_GAIN = 0.003   # guadagno minimo di R^2 OOT per accettare una feature (early stop)
MAX_FEATURES = 14  # tetto di sicurezza alla lunghezza della ricetta

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
    reg_lambda=2.0, random_state=42, n_jobs=4,
)


def input_pool(tf: pd.DataFrame, kind: str = "all") -> list[str]:
    """Le feature-GIOCATORE candidate. Esclude chiavi, esiti, z-epoca.

    kind:
      'all'       -> tutte (struttura + talento)
      'structure' -> SOLO caratteristiche strutturali costruibili (no livello/
                     impatto) -> la ricetta non-tautologica
      'talent'    -> SOLO le feature di talento (servono come CONTROLLO per
                     calcolare l'over-performance)
    """
    num = tf.select_dtypes("number").columns.tolist()
    pool = [c for c in num
            if c not in OUTCOME_COLS and c not in KEY_COLS
            and not c.startswith("zera_")]
    if kind == "structure":
        return [c for c in pool if not is_talent(c)]
    if kind == "talent":
        return [c for c in pool if is_talent(c)]
    return pool


def _oot_r2(data: pd.DataFrame, cols: list[str], target: str) -> float:
    """R^2 OUT-OF-TIME: walk-forward su [OOT_START, OOT_END]. Per ogni anno
    alleno su tutte le stagioni precedenti e predico quell'anno; poi un solo R^2
    sull'insieme delle predizioni out-of-time (onesto, nessun leakage)."""
    y_true, y_pred = [], []
    for year in range(OOT_START, OOT_END + 1):
        tr = data[data["SEASON_START_YEAR"] < year]
        te = data[data["SEASON_START_YEAR"] == year]
        if len(te) == 0 or len(tr) < 50:
            continue
        m = XGBRegressor(**XGB_PARAMS)
        m.fit(tr[cols], tr[target])
        y_true.append(te[target].to_numpy())
        y_pred.append(m.predict(te[cols]))
    if not y_true:
        return float("nan")
    return float(r2_score(np.concatenate(y_true), np.concatenate(y_pred)))


def _too_correlated(cand: str, kept: list[str], corr: pd.DataFrame) -> bool:
    return any(corr.loc[cand, k] > CORR_THRESHOLD for k in kept)


def greedy_recipe(data: pd.DataFrame, pool: list[str], target: str) -> dict:
    """Forward-selection scorrelata guidata dall'R^2 out-of-time."""
    corr = data[pool].corr().abs()
    kept: list[str] = []
    curve = []          # R^2 OOT cumulato passo per passo
    best_r2 = -np.inf

    while len(kept) < MAX_FEATURES:
        # a ogni passo si sceglie il candidato che porta al R^2 OUT-OF-TIME
        # ASSOLUTO piu' alto (non il "delta" da una baseline: al primo passo la
        # baseline sarebbe -inf e farebbe vincere la prima feature iterata).
        best_feat, best_new_r2 = None, -np.inf
        for cand in pool:
            if cand in kept or _too_correlated(cand, kept, corr):
                continue
            r2 = _oot_r2(data, kept + [cand], target)
            if np.isnan(r2):
                continue
            if r2 > best_new_r2:
                best_feat, best_new_r2 = cand, r2
        if best_feat is None:
            break
        gain = best_new_r2 - best_r2
        # early stop: il candidato migliore non aggiunge abbastanza spiegazione
        # (vale solo dal 2o passo in poi; al 1o best_r2=-inf => si entra sempre)
        if kept and gain < MIN_GAIN:
            break
        kept.append(best_feat)
        best_r2 = best_new_r2
        curve.append({"feature": best_feat, "r2_oot": round(best_r2, 4),
                      "gain": round(gain, 4) if np.isfinite(gain) else None})
        print(f"  +{best_feat:28s}  R2_oot={best_r2:.4f}  (+{gain:.4f})")

    return {"target": target, "selected": kept, "n_selected": len(kept),
            "r2_oot_final": round(best_r2, 4), "curve": curve}


def _shap_direction(data: pd.DataFrame, cols: list[str], target: str) -> dict:
    """Direzione (segno) di ogni leva della ricetta, via correlazione tra valore
    e SHAP (riusa il pattern di what_builds_rating.py). Modello allenato su TUTTO
    (qui serve solo la direzione interpretativa, non una metrica predittiva)."""
    import shap
    m = XGBRegressor(**XGB_PARAMS)
    m.fit(data[cols], data[target])
    sv = pd.DataFrame(shap.TreeExplainer(m).shap_values(data[cols]), columns=cols)
    out = {}
    for c in cols:
        sign = 0.0
        if data[c].std() > 0 and sv[c].std() > 0:
            sign = float(np.sign(np.corrcoef(data[c], sv[c])[0, 1]))
        out[c] = {"importance": round(float(sv[c].abs().mean()), 4), "direction": sign}
    return out


def factor_of(feat: str) -> str:
    """Mappa una feature sull'ipotesi-tesi corrispondente, per rispondere:
    'quante delle feature vincenti sono atletismo/tiro/playmaking diffuso?'."""
    for t, name in [("T1", "T1 homegrown"), ("T2", "T2 size/athleticism"),
                    ("T3", "T3 playoff exp"), ("T4", "T4 three-point"),
                    ("T5", "T5 spread playmaking")]:
        if f"{t}_" in feat:
            return name
    if feat.startswith("BAND_"):
        if any(k in feat for k in ("height", "wingspan", "vertical")):
            return "T2 size/athleticism"   # fisico per ruolo
        if "3p" in feat:
            return "T4 three-point"        # tiro per ruolo
        if "po_depth" in feat:
            return "T3 playoff exp"        # esperienza per ruolo
        return "ROLE composition"
    if "star" in feat:
        return "ROSTER star power"
    if feat == "PACE":
        return "PACE / style"
    return "OTHER"


def _factor_summary(direction: dict) -> dict:
    """Somma dell'importanza per fattore: la 'ricetta' tradotta nelle 5 ipotesi."""
    agg: dict[str, float] = {}
    for feat, d in direction.items():
        agg[factor_of(feat)] = agg.get(factor_of(feat), 0.0) + d["importance"]
    return {k: round(v, 4) for k, v in sorted(agg.items(), key=lambda x: -x[1])}


def run(targets: tuple[str, ...] = ("NET_RATING", "W_PCT")) -> dict:
    tf = pd.read_parquet(PROCESSED / "team_season_features.parquet")
    pool = input_pool(tf)
    results = {"input_pool_size": len(pool), "corr_threshold": CORR_THRESHOLD,
               "oot_window": [OOT_START, OOT_END]}

    for target in targets:
        data = tf.dropna(subset=[target]).copy()
        data[pool] = data[pool].fillna(data[pool].median())
        print("\n" + "=" * 64)
        print(f"RICETTA per {target}  (pool={len(pool)} feature, criterio R2 OUT-OF-TIME)")
        print("=" * 64)
        rec = greedy_recipe(data, pool, target)
        rec["direction"] = _shap_direction(data, rec["selected"], target)
        rec["by_factor"] = _factor_summary(rec["direction"])
        results[target] = rec
        _report(rec)

    (PROCESSED / "recipe.json").write_text(json.dumps(results, indent=2))
    print(f"\nSalvato data/processed/recipe.json")
    return results


def _report(rec: dict) -> None:
    print(f"\n>>> {rec['n_selected']} feature scorrelate spiegano R2_oot="
          f"{rec['r2_oot_final']} del {rec['target']}")
    print("\nLa ricetta (con direzione):")
    for f in rec["selected"]:
        d = rec["direction"][f]
        arrow = "↑ alza" if d["direction"] > 0 else ("↓ abbassa" if d["direction"] < 0 else "·")
        print(f"  {f:28s} imp={d['importance']:.3f}  {arrow}   [{factor_of(f)}]")
    print("\nContributo per IPOTESI (T1..T5 + ruolo/star/pace):")
    for k, v in rec["by_factor"].items():
        print(f"  {k:24s} {v:.3f}")


if __name__ == "__main__":
    run()
