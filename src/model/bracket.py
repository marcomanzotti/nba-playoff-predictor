"""Bracket per CONFERENCE + simulatore Monte Carlo (Fase 4).

Ricostruzione PRECISA del tabellone usando conference e seed ufficiali
(da team_conference.parquet):
  - 8 squadre per conference, accoppiate 1v8 / 2v7 / 3v6 / 4v5 al 1o turno;
  - 2o turno: vincitori (1v8 vs 4v5) e (2v7 vs 3v6) -> 4 semifinali conf;
  - finali di conference: intra-conference;
  - NBA Finals: vincente Est vs vincente Ovest.

Le 8 qualificate per conference sono le squadre che hanno effettivamente
disputato i playoff quell'anno (robusto al play-in: prendiamo chi compare nel
1o turno reale, ordinandole per seed ufficiale).

Il simulatore usa il modello di serie per estrarre i vincitori e, ripetendo N
volte, stima per ogni squadra P(2o turno), P(finale conf), P(Finals), P(titolo).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"

# accoppiamenti standard del 1o turno per seed (1-based)
FIRST_ROUND_PAIRS = [(1, 8), (4, 5), (3, 6), (2, 7)]  # ordine = posizione nel bracket


@dataclass
class Node:
    round: int
    team_a: str | None = None
    team_b: str | None = None
    child_a: "Node | None" = None
    child_b: "Node | None" = None


def _conf_seeds(year: int) -> dict[str, list[str]]:
    """Per ogni conference, la lista delle 8 qualificate ordinate per seed.

    Qualificate = squadre presenti nel 1o turno reale di quell'anno (gestisce il
    play-in: contano gli 8 che hanno davvero iniziato i playoff).
    """
    tc = pd.read_parquet(INTERIM / "team_conference.parquet")
    series = pd.read_parquet(INTERIM / "playoff_series.parquet")
    yr_series = series[(series["SEASON_START_YEAR"] == year) & (series["ROUND"] == 1)]
    qualified = set(yr_series["TEAM_A"]) | set(yr_series["TEAM_B"])

    tc_y = tc[(tc["SEASON_START_YEAR"] == year) & (tc["TEAM_ABBREVIATION"].isin(qualified))]
    out = {}
    for conf, g in tc_y.groupby("CONFERENCE"):
        ordered = g.sort_values("SEED")["TEAM_ABBREVIATION"].tolist()
        out[conf] = ordered
    return out


def build_conference_bracket(year: int) -> Node:
    """Costruisce l'albero del bracket dal seeding ufficiale per conference."""
    conf_seeds = _conf_seeds(year)
    conf_finals_nodes = []
    for conf, seeds in conf_seeds.items():
        if len(seeds) < 8:
            # stagione con dati parziali: completiamo col disponibile in ordine
            seeds = (seeds + seeds)[:8]
        # foglie 1o turno
        r1 = []
        for hi, lo in FIRST_ROUND_PAIRS:
            r1.append(Node(round=1, team_a=seeds[hi - 1], team_b=seeds[lo - 1]))
        # 2o turno: (1v8 vs 4v5), (3v6 vs 2v7)
        r2a = Node(round=2, child_a=r1[0], child_b=r1[1])
        r2b = Node(round=2, child_a=r1[2], child_b=r1[3])
        # finale di conference
        cf = Node(round=3, child_a=r2a, child_b=r2b)
        conf_finals_nodes.append(cf)

    # NBA Finals: le due finali di conference
    if len(conf_finals_nodes) == 2:
        return Node(round=4, child_a=conf_finals_nodes[0], child_b=conf_finals_nodes[1])
    # fallback (non dovrebbe accadere nell'orizzonte)
    return conf_finals_nodes[0]


def _postorder(root: Node) -> list[Node]:
    out = []
    def visit(n):
        if n is None:
            return
        if n.round > 1:
            visit(n.child_a)
            visit(n.child_b)
        out.append(n)
    visit(root)
    return out


def _all_teams(root: Node) -> list[str]:
    teams = set()
    def visit(n):
        if n is None:
            return
        if n.round == 1:
            teams.update([n.team_a, n.team_b])
        else:
            visit(n.child_a)
            visit(n.child_b)
    visit(root)
    return sorted(teams)


def _resolve_teams(node: Node, sim_winner: dict) -> tuple[str, str]:
    if node.round == 1:
        return node.team_a, node.team_b
    return sim_winner[id(node.child_a)], sim_winner[id(node.child_b)]


def simulate_bracket(root: Node, predictor, year: int, n_sims: int = 5000,
                     rng: np.random.Generator | None = None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(42)
    nodes_post = _postorder(root)
    teams = _all_teams(root)
    reach = {t: {2: 0, 3: 0, 4: 0, "champ": 0} for t in teams}

    for _ in range(n_sims):
        sim_winner: dict[int, str] = {}
        for node in nodes_post:
            a, b = _resolve_teams(node, sim_winner)
            pa = predictor.prob(year, a, b)
            w = a if rng.random() < pa else b
            sim_winner[id(node)] = w
            nxt = node.round + 1
            if nxt in (2, 3, 4):
                reach[w][nxt] += 1
            if node.round == 4:
                reach[w]["champ"] += 1

    rows = []
    for t in teams:
        rows.append({
            "TEAM": t,
            "P_R2": reach[t][2] / n_sims,
            "P_CONF_FINALS": reach[t][3] / n_sims,
            "P_FINALS": reach[t][4] / n_sims,
            "P_CHAMPION": reach[t]["champ"] / n_sims,
        })
    return pd.DataFrame(rows).sort_values("P_CHAMPION", ascending=False).reset_index(drop=True)


def most_likely_bracket(root: Node, predictor, year: int) -> list[dict]:
    """Bracket 'modale': ad ogni serie avanza la squadra con prob > 0.5.
    Utile per confrontare il bracket PREDETTO con quello reale (backtest)."""
    sim_winner: dict[int, str] = {}
    picks = []
    for node in _postorder(root):
        a, b = _resolve_teams(node, sim_winner)
        pa = predictor.prob(year, a, b)
        w = a if pa >= 0.5 else b
        sim_winner[id(node)] = w
        picks.append({"round": node.round, "team_a": a, "team_b": b,
                      "pred_winner": w, "p_a": round(pa, 3)})
    return picks
