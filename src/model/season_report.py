"""Report completo di una stagione di playoff (richiesta utente #1).

Produce DUE viste complementari:

 A) PROBABILITA' PER SQUADRA (advancement): per ogni squadra ai playoff,
    la probabilita' di superare OGNI turno e di vincere il titolo:
       P(passa 1o turno), P(passa 2o), P(passa finale conf), P(vince titolo)
    Stimate via Monte Carlo del bracket (tutti i possibili tabelloni).

 B) PROBABILITA' PER MATCHUP (head-to-head di serie): per OGNI confronto
    effettivamente in palio in un dato round, la % di vittoria della serie
    di ENTRAMBE le squadre  ->  P(A passa) e P(B passa) = 1 - P(A passa).
    Include sia i matchup gia' noti (1o turno) sia, opzionalmente, i piu'
    probabili dei round successivi.

Output (data/processed/):
  season_report_{year}_advancement.parquet   (vista A)
  season_report_{year}_matchups.parquet       (vista B)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.model.bracket import (
    Node,
    _all_teams,
    _postorder,
    build_conference_bracket,
)
from src.season_labels import season_label, title_year

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"

ROUND_NAMES = {1: "1o turno", 2: "2o turno", 3: "finale conference", 4: "NBA Finals"}


def advancement_probabilities(root: Node, predictor, year: int,
                              n_sims: int = 10000,
                              rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Vista A: per ogni squadra, P(supera 1o/2o/finale conf) e P(titolo).

    'Supera il turno R' = vince la serie del turno R (e quindi avanza).
    """
    rng = rng or np.random.default_rng(year)
    nodes_post = _postorder(root)
    teams = _all_teams(root)
    # 'win[r]' = quante volte la squadra ha VINTO la serie del round r
    win = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in teams}

    for _ in range(n_sims):
        sim_winner: dict[int, str] = {}
        for node in nodes_post:
            if node.round == 1:
                a, b = node.team_a, node.team_b
            else:
                a, b = sim_winner[id(node.child_a)], sim_winner[id(node.child_b)]
            pa = predictor.prob(year, a, b)
            w = a if rng.random() < pa else b
            sim_winner[id(node)] = w
            win[w][node.round] += 1

    rows = []
    for t in teams:
        rows.append({
            "SEASON": season_label(year), "TITLE_YEAR": title_year(year), "TEAM": t,
            "P_pass_R1": win[t][1] / n_sims,
            "P_pass_R2": win[t][2] / n_sims,
            "P_pass_ConfFinals": win[t][3] / n_sims,
            "P_champion": win[t][4] / n_sims,
        })
    df = pd.DataFrame(rows).sort_values("P_champion", ascending=False).reset_index(drop=True)
    return df


def matchup_probabilities(root: Node, predictor, year: int) -> pd.DataFrame:
    """Vista B: per ogni serie del bracket, la % di vittoria di ENTRAMBE.

    Per il 1o turno i matchup sono certi. Per i round successivi mostriamo il
    matchup 'modale' (avversari piu' probabili) con la relativa percentuale,
    etichettato come ipotetico.
    """
    rows = []
    # 1o turno: matchup certi
    for node in _postorder(root):
        if node.round != 1:
            continue
        a, b = node.team_a, node.team_b
        pa = predictor.prob(year, a, b)
        rows.append(_matchup_row(year, node.round, a, b, pa, certain=True))

    # round successivi: usiamo gli avversari modali (bracket a prob>0.5)
    sim_winner: dict[int, str] = {}
    for node in _postorder(root):
        if node.round == 1:
            a, b = node.team_a, node.team_b
        else:
            a, b = sim_winner[id(node.child_a)], sim_winner[id(node.child_b)]
        pa = predictor.prob(year, a, b)
        sim_winner[id(node)] = a if pa >= 0.5 else b
        if node.round > 1:
            rows.append(_matchup_row(year, node.round, a, b, pa, certain=False))

    return pd.DataFrame(rows)


def _matchup_row(year, rnd, a, b, pa, certain) -> dict:
    return {
        "SEASON": season_label(year), "TITLE_YEAR": title_year(year),
        "ROUND": rnd, "ROUND_NAME": ROUND_NAMES[rnd],
        "TEAM_A": a, "TEAM_B": b,
        "P_A_wins_series": round(float(pa), 3),
        "P_B_wins_series": round(float(1 - pa), 3),
        "matchup_certain": certain,
    }


def build_season_report(predictor, year: int, n_sims: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = build_conference_bracket(year)
    adv = advancement_probabilities(root, predictor, year, n_sims=n_sims)
    mu = matchup_probabilities(root, predictor, year)
    adv.to_parquet(PROCESSED / f"season_report_{year}_advancement.parquet", index=False)
    mu.to_parquet(PROCESSED / f"season_report_{year}_matchups.parquet", index=False)
    return adv, mu
