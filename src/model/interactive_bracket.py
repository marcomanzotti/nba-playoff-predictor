"""Interactive bracket: a what-if engine on top of the conference bracket.

The base bracket (`src.model.bracket`) builds the official tree from seeds and
simulates it with the series model. This module adds two things the dashboard
needs:

  1. **Stable series IDs.** Each series node gets a deterministic string id (e.g.
     "E-R1-0", "W-R2-1", "FINALS") so the UI can reference a specific series
     across Streamlit reruns and store user overrides keyed by it. `id()` is not
     stable across runs, so we never expose it.

  2. **What-if overrides.** The user can *force the winner* of any series
     ("what if Atlanta had beaten New York?"). We then recompute everything
     downstream of that pick, both as a single most-likely bracket and as
     Monte-Carlo title odds **conditioned** on the forced results.

Public API
----------
    bracket = InteractiveBracket(predictor, year)
    bracket.series()                  -> list[SeriesView] (modal bracket, no force)
    bracket.series(overrides={...})   -> modal bracket given forced winners
    bracket.title_odds(overrides)     -> DataFrame P(champion|overrides) per team

`overrides` is a dict {series_id: forced_winner_abbr}. A forced winner must be one
of the two teams that actually reach that series given the *upstream* overrides;
an override on a series whose matchup changed upstream is silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model.bracket import Node, build_conference_bracket
from src.season_labels import season_label, title_year

ROUND_NAME = {1: "First round", 2: "Conference semifinals",
              3: "Conference finals", 4: "NBA Finals"}


@dataclass
class SeriesView:
    """One series as the UI sees it."""
    sid: str                 # stable id, e.g. "E-R1-0"
    round: int               # 1..4
    round_name: str
    conference: str | None   # "East"/"West" for rounds 1-3, None for the Finals
    team_a: str | None       # may be None if upstream not yet decided
    team_b: str | None
    p_a: float               # model P(team_a wins the series)
    p_b: float
    modal_winner: str | None # who advances in the most-likely bracket
    forced_winner: str | None  # set if this series is overridden (and valid)


class InteractiveBracket:
    def __init__(self, predictor, year: int):
        self.predictor = predictor
        self.year = year
        self.root = build_conference_bracket(year)
        # assign stable ids + conference labels by walking the tree
        self._sid: dict[int, str] = {}
        self._conf: dict[int, str | None] = {}
        self._label_tree()

    # ----- id / labelling -------------------------------------------------
    def _label_tree(self):
        """Give each node a stable id and conference tag.

        The root (round 4) is the Finals. Its two children are the East and West
        conference-final nodes. We tag a whole conference subtree with its label.
        """
        # children of the finals = the two conference brackets
        kids = [self.root.child_a, self.root.child_b]
        conf_labels = self._infer_conf_labels(kids)
        self._sid[id(self.root)] = "FINALS"
        self._conf[id(self.root)] = None
        for child, label in zip(kids, conf_labels):
            self._label_conf_subtree(child, label)

    def _infer_conf_labels(self, conf_nodes: list[Node]) -> list[str]:
        """Tag each conference subtree as East/West from its teams' conference."""
        from pathlib import Path
        interim = Path(__file__).resolve().parents[2] / "data" / "interim"
        tc = pd.read_parquet(interim / "team_conference.parquet")
        tc_y = tc[tc["SEASON_START_YEAR"] == self.year]
        conf_of = dict(zip(tc_y["TEAM_ABBREVIATION"], tc_y["CONFERENCE"]))
        labels = []
        for node in conf_nodes:
            teams = self._leaf_teams(node)
            confs = [conf_of.get(t) for t in teams if conf_of.get(t)]
            # majority vote (robust to any odd data)
            label = max(set(confs), key=confs.count) if confs else "?"
            labels.append(label)
        # guarantee distinct East/West even if data is messy
        if len(set(labels)) == 1:
            labels = ["East", "West"]
        return labels

    @staticmethod
    def _leaf_teams(node: Node) -> list[str]:
        out = []
        def visit(n):
            if n is None:
                return
            if n.round == 1:
                out.extend([n.team_a, n.team_b])
            else:
                visit(n.child_a)
                visit(n.child_b)
        visit(node)
        return out

    def _label_conf_subtree(self, conf_root: Node, label: str):
        short = "E" if str(label).lower().startswith("e") else "W"
        # round 1 slots numbered by position
        r1_counter = [0]
        r2_counter = [0]

        def visit(n: Node):
            if n is None:
                return
            self._conf[id(n)] = label
            if n.round == 1:
                self._sid[id(n)] = f"{short}-R1-{r1_counter[0]}"
                r1_counter[0] += 1
            elif n.round == 2:
                visit(n.child_a)
                visit(n.child_b)
                self._sid[id(n)] = f"{short}-R2-{r2_counter[0]}"
                r2_counter[0] += 1
            elif n.round == 3:
                visit(n.child_a)
                visit(n.child_b)
                self._sid[id(n)] = f"{short}-CF"
        # ensure children get visited/labelled before the parent for round 1
        # (visit handles ordering via recursion)
        visit(conf_root)

    # ----- resolution with overrides -------------------------------------
    def _resolve(self, overrides: dict[str, str] | None):
        """Walk the tree bottom-up; return per-node resolved (a, b, winner).

        `winner` is the forced winner if a valid override exists for that series,
        otherwise the modal winner (higher model probability). A forced winner is
        only honored if it's actually one of the two teams reaching the series.
        """
        overrides = overrides or {}
        resolved: dict[int, tuple[str | None, str | None, str | None]] = {}

        def teams_of(node: Node) -> tuple[str | None, str | None]:
            if node.round == 1:
                return node.team_a, node.team_b
            wa = resolved[id(node.child_a)][2]
            wb = resolved[id(node.child_b)][2]
            return wa, wb

        def visit(node: Node):
            if node is None:
                return
            if node.round > 1:
                visit(node.child_a)
                visit(node.child_b)
            a, b = teams_of(node)
            sid = self._sid[id(node)]
            forced = overrides.get(sid)
            if a is None or b is None:
                resolved[id(node)] = (a, b, None)
                return
            if forced in (a, b):
                winner = forced
            else:
                p = self.predictor.prob(self.year, a, b)
                winner = a if p >= 0.5 else b
            resolved[id(node)] = (a, b, winner)

        visit(self.root)
        return resolved

    def series(self, overrides: dict[str, str] | None = None) -> list[SeriesView]:
        """All series as SeriesView, in display order (East R1..CF, West R1..CF,
        Finals), with teams resolved given the overrides."""
        overrides = overrides or {}
        resolved = self._resolve(overrides)
        views: list[SeriesView] = []

        def make(node: Node) -> SeriesView:
            a, b, winner = resolved[id(node)]
            if a is not None and b is not None:
                p_a = self.predictor.prob(self.year, a, b)
            else:
                p_a = 0.5
            sid = self._sid[id(node)]
            forced = overrides.get(sid)
            forced = forced if forced in (a, b) else None
            return SeriesView(
                sid=sid, round=node.round, round_name=ROUND_NAME[node.round],
                conference=self._conf[id(node)], team_a=a, team_b=b,
                p_a=round(float(p_a), 4), p_b=round(float(1 - p_a), 4),
                modal_winner=winner, forced_winner=forced,
            )

        # collect every node, order by (conf, round, slot)
        all_nodes: list[Node] = []
        def collect(n: Node):
            if n is None:
                return
            if n.round > 1:
                collect(n.child_a)
                collect(n.child_b)
            all_nodes.append(n)
        collect(self.root)

        def sort_key(n: Node):
            sid = self._sid[id(n)]
            if sid == "FINALS":
                return (2, 9, 0)
            conf_rank = 0 if sid.startswith("E") else 1
            if "R1" in sid:
                return (conf_rank, 1, int(sid.split("-")[-1]))
            if "R2" in sid:
                return (conf_rank, 2, int(sid.split("-")[-1]))
            return (conf_rank, 3, 0)  # conf finals

        for n in sorted(all_nodes, key=sort_key):
            views.append(make(n))
        return views

    def title_odds(self, overrides: dict[str, str] | None = None,
                   n_sims: int = 5000,
                   rng: np.random.Generator | None = None) -> pd.DataFrame:
        """Monte-Carlo title / advancement odds **conditioned** on overrides.

        Forced series always resolve to the forced winner; every other series is
        simulated from the model probability. Returns per-team probabilities of
        passing each round and winning the title, sorted by title odds.
        """
        overrides = overrides or {}
        rng = rng or np.random.default_rng(self.year)
        # postorder list of nodes for simulation
        post: list[Node] = []
        def collect(n: Node):
            if n is None:
                return
            if n.round > 1:
                collect(n.child_a)
                collect(n.child_b)
            post.append(n)
        collect(self.root)

        teams = set(self._leaf_teams(self.root))
        win = {t: {1: 0, 2: 0, 3: 0, 4: 0} for t in teams}

        for _ in range(n_sims):
            sim_winner: dict[int, str] = {}
            for node in post:
                if node.round == 1:
                    a, b = node.team_a, node.team_b
                else:
                    a, b = sim_winner[id(node.child_a)], sim_winner[id(node.child_b)]
                sid = self._sid[id(node)]
                forced = overrides.get(sid)
                if forced in (a, b):
                    w = forced
                else:
                    pa = self.predictor.prob(self.year, a, b)
                    w = a if rng.random() < pa else b
                sim_winner[id(node)] = w
                win[w][node.round] += 1

        rows = []
        for t in teams:
            rows.append({
                "SEASON": season_label(self.year),
                "TITLE_YEAR": title_year(self.year),
                "TEAM": t,
                "P_pass_R1": win[t][1] / n_sims,
                "P_pass_R2": win[t][2] / n_sims,
                "P_pass_ConfFinals": win[t][3] / n_sims,
                "P_champion": win[t][4] / n_sims,
            })
        return (pd.DataFrame(rows)
                .sort_values("P_champion", ascending=False)
                .reset_index(drop=True))
