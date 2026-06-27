"""Ingest a NEW season from JSON and predict its playoffs automatically.

Intended workflow: at the end of a regular season, drop a JSON with per-player
and per-team values; this runs the full feature pipeline (minutes weighting →
roles → T1-T5 → team aggregation → series model → bracket) and produces the
playoff predictions for that season — no manual steps.

JSON schema (see data/sample_new_season.json for a concrete example):
{
  "season_start_year": 2026,
  "players": [
     {"PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE",
      "GP","MIN","PTS","REB","AST","STL","BLK","TOV","PF",
      "FG3M","FG3A","FG3_PCT","FGA","FTA",
      "TS_PCT","USG_PCT","AST_PCT","REB_PCT","OREB_PCT","DREB_PCT",
      "NET_RATING","PIE","MIN_TOT","PTS_TOT","FG3M_TOT","FG3A_TOT",
      # career/physical (optional — proxied if missing):
      "HOMETOWN_SCORE","SEASONS_EXP","PLAYOFF_DEPTH_PRIOR","DEEP_RUNS_PRIOR","TITLES_PRIOR",
      "HEIGHT_WO_SHOES","WINGSPAN","MAX_VERTICAL_LEAP","LANE_AGILITY_TIME",
      "LEVEL"   # optional: superstar/all_star/... else inferred
     }, ...
  ],
  "teams": [
     {"TEAM_ID","TEAM_ABBREVIATION","CONFERENCE","SEED",
      "W","L","W_PCT","OFF_RATING","DEF_RATING","NET_RATING","PACE","TS_PCT","EFG_PCT",
      "FG3M","FG3A","FG3_PCT","PTS",
      "HOME_WIN_PCT","AWAY_WIN_PCT"}, ...
  ],
  "first_round": [   # the 8 first-round matchups (4 per conference)
     {"TEAM_A","TEAM_B"}, ...
  ],
  "head_to_head": [  # optional regular-season H2H
     {"TEAM","OPP","WINS"}, ...
  ]
}

Only the fields needed by the model are required; the rest are proxied/defaulted.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"


def _build_team_features_from_json(payload: dict) -> pd.DataFrame:
    """Aggregate per-player rows into team features (minutes-weighted), then
    merge official team stats. Mirrors src/features/team_features.py at runtime."""
    from src.features.team_features import (_role_band_features, _shannon_entropy,
                                           _wmean)

    year = int(payload["season_start_year"])
    players = pd.DataFrame(payload["players"])
    teams = pd.DataFrame(payload["teams"])

    # ensure required derived columns exist with sane defaults
    for col, default in [("HOMETOWN_SCORE", 0), ("PLAYOFF_DEPTH_PRIOR", 0.0),
                         ("DEEP_RUNS_PRIOR", 0), ("TITLES_PRIOR", 0), ("SEASONS_EXP", 0),
                         ("MIN_TOT", None), ("LEVEL", "role_player"), ("ROLE_BAND", "wing")]:
        if col not in players.columns:
            players[col] = default
    if players["MIN_TOT"].isna().all():
        players["MIN_TOT"] = players["MIN"] * players.get("GP", 1)
    for c in ("FG3M_TOT", "FG3A_TOT", "PTS_TOT"):
        if c not in players.columns:
            base = c.replace("_TOT", "")
            players[c] = players.get(base, 0) * players.get("GP", 1)
    # infer ROLE_BAND if absent
    if (players["ROLE_BAND"] == "wing").all() and "HEIGHT_WO_SHOES" in players.columns:
        players = _infer_roles(players)
    # infer LEVEL (+ LEVEL_ORD) if absent
    players = _infer_levels(players)

    # guarantee every column the aggregation reads exists (proxy/default missing)
    for col in ("FG3A", "FG3_PCT", "HEIGHT_WO_SHOES", "WINGSPAN",
                "MAX_VERTICAL_LEAP", "PLAYOFF_DEPTH_PRIOR", "AST_PCT", "REB_PCT", "BLK"):
        if col not in players.columns:
            players[col] = np.nan

    w = "MIN_TOT"
    rows = []
    for team_id, g in players.groupby("TEAM_ID"):
        tot = g[w].sum()
        if tot <= 0:
            continue
        min_share = g[w] / tot
        pts_tot = g["PTS_TOT"].fillna(0)
        pts_share = pts_tot / pts_tot.sum() if pts_tot.sum() > 0 else min_share
        hs = g["HOMETOWN_SCORE"].fillna(0)
        is_star = g["LEVEL"].isin(["superstar", "all_star"])

        row = {
            "SEASON_START_YEAR": year, "TEAM_ID": team_id,
            "TEAM_ABBREVIATION": g["TEAM_ABBREVIATION"].iloc[0],
            "T1_hometown_minshare": float((min_share * (hs / 4)).sum()),
            "T1_homegrown_core": float((min_share * (hs >= 3).astype(int)).sum()),
            "T2_height": _wmean(g.get("HEIGHT_WO_SHOES", pd.Series(np.nan, index=g.index)), g[w]),
            "T2_wingspan": _wmean(g.get("WINGSPAN", pd.Series(np.nan, index=g.index)), g[w]),
            "T2_vertical": _wmean(g.get("MAX_VERTICAL_LEAP", pd.Series(np.nan, index=g.index)), g[w]),
            "T3_playoff_depth": float((min_share * g["PLAYOFF_DEPTH_PRIOR"].fillna(0)).sum()),
            "T3_deep_runs": float((min_share * g["DEEP_RUNS_PRIOR"].fillna(0)).sum()),
            "T3_titles": float((min_share * g["TITLES_PRIOR"].fillna(0)).sum()),
            "T4_3p_pct": float(g["FG3M_TOT"].sum() / max(g["FG3A_TOT"].sum(), 1)),
            "T4_n_shooters": int(((g.get("FG3A_TOT", 0) >= 100) & (g.get("FG3_PCT", 0) >= 0.35)).sum()),
            "T5_nonstar_min_share": float(min_share[~is_star].sum()),
            "T5_nonstar_pts_share": float(pts_share[~is_star].sum()),
            "T5_pts_entropy": _shannon_entropy(pts_share.to_numpy()),
            "T5_min_entropy": _shannon_entropy(min_share.to_numpy()),
            "n_superstars": int((g["LEVEL"] == "superstar").sum()),
            "n_allstars": int((g["LEVEL"] == "all_star").sum()),
        }
        row.update(_role_band_features(g, w))
        rows.append(row)

    tp = pd.DataFrame(rows)
    # merge official team stats from JSON
    team_cols = ["TEAM_ID", "W", "L", "W_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING",
                 "PACE", "TS_PCT", "EFG_PCT", "FG3M", "FG3A", "FG3_PCT", "PTS",
                 "HOME_WIN_PCT", "AWAY_WIN_PCT"]
    avail = [c for c in team_cols if c in teams.columns]
    tp = tp.merge(teams[avail], on="TEAM_ID", how="left")
    return tp.fillna(tp.median(numeric_only=True))


def _infer_roles(players: pd.DataFrame) -> pd.DataFrame:
    from src.features.player_role import _band_from_signals
    h = players["HEIGHT_WO_SHOES"]
    lo, hi = h.quantile(0.33), h.quantile(0.66)
    players["ROLE_BAND"] = [
        _band_from_signals(hh, a, r, b, lo, hi)
        for hh, a, r, b in zip(h, players.get("AST_PCT"), players.get("REB_PCT"), players.get("BLK"))
    ]
    return players


LEVELS = ["bench_warmer", "bench", "role_player", "quality_starter", "all_star", "superstar"]


def _infer_levels(players: pd.DataFrame) -> pd.DataFrame:
    have_levels = ("LEVEL" in players.columns and players["LEVEL"].notna().any()
                   and not (players["LEVEL"] == "role_player").all())
    if not have_levels:
        # simple impact score from PIE/USG/MIN if available
        score = players.get("PIE", 0) * 2 + players.get("USG_PCT", 0) + players.get("MIN", 0) / 10
        pct = score.rank(pct=True)
        def lv(p):
            return ("superstar" if p >= .97 else "all_star" if p >= .90 else
                    "quality_starter" if p >= .70 else "role_player" if p >= .40 else
                    "bench" if p >= .15 else "bench_warmer")
        players["LEVEL"] = pct.apply(lv)
    # ordinal needed by role-band features
    players["LEVEL_ORD"] = players["LEVEL"].map({lv: i for i, lv in enumerate(LEVELS)}).fillna(2)
    return players


def predict_new_season(payload: dict, n_sims: int = 5000) -> dict:
    """Full pipeline: JSON → team features → series model → bracket predictions."""
    from src.model.backtest_bracket import _fit_until
    from src.model.bracket import Node
    from src.model.season_report import (advancement_probabilities,
                                        matchup_probabilities)
    from src.model.series_predictor import SeriesPredictor, feature_columns
    from src.model.walkforward import _feature_cols

    year = int(payload["season_start_year"])
    tp = _build_team_features_from_json(payload)

    # train the series model on ALL historical data (everything we have)
    feat = _feature_cols(pd.read_parquet(PROCESSED / "series_dataset.parquet"))
    model = _fit_until(year, feat)

    # build a predictor that reads from OUR new team-features instead of the DB
    predictor = _NewSeasonPredictor(model, feat, tp, payload)
    root = _bracket_from_payload(payload)

    adv = advancement_probabilities(root, predictor, year, n_sims=n_sims)
    mu = matchup_probabilities(root, predictor, year)
    return {"advancement": adv, "matchups": mu, "year": year}


class _NewSeasonPredictor:
    """SeriesPredictor variant backed by the uploaded team features."""
    def __init__(self, model, feat_order, team_features, payload):
        from src.model.series_predictor import feature_columns
        self.model = model
        self.feat_order = feat_order
        self.tf = team_features
        self.team_cols = feature_columns(self.tf)
        self.tf_idx = self.tf.set_index("TEAM_ABBREVIATION")
        self.seed = {t["TEAM_ABBREVIATION"]: t.get("SEED", 99)
                     for t in payload["teams"] if "TEAM_ABBREVIATION" in t}
        self.h2h = {(h["TEAM"], h["OPP"]): h["WINS"] for h in payload.get("head_to_head", [])}
        self._cache = {}

    def prob(self, year, a, b):
        key = (a, b)
        if key in self._cache:
            return self._cache[key]
        if a not in self.tf_idx.index or b not in self.tf_idx.index:
            return 0.5
        fa, fb = self.tf_idx.loc[a, self.team_cols], self.tf_idx.loc[b, self.team_cols]
        if isinstance(fa, pd.DataFrame):
            fa = fa.iloc[0]
        if isinstance(fb, pd.DataFrame):
            fb = fb.iloc[0]
        diff = (fa - fb).astype(float)
        row = {f"d_{c}": diff[c] for c in self.team_cols}
        row["H2H_DIFF"] = self.h2h.get((a, b), 0) - self.h2h.get((b, a), 0)
        row["HOME_COURT"] = 1.0 if self.seed.get(a, 99) < self.seed.get(b, 99) else -1.0
        x = pd.DataFrame([row]).reindex(columns=self.feat_order, fill_value=0.0).fillna(0.0)
        xv = x.to_numpy()
        p = float((self.model.predict_proba(xv)[0, 1] + (1 - self.model.predict_proba(-xv)[0, 1])) / 2)
        self._cache[key] = p
        self._cache[(b, a)] = 1 - p
        return p


def _bracket_from_payload(payload):
    """Build the conference bracket tree from the provided first-round matchups."""
    from src.model.bracket import Node
    conf_of = {t["TEAM_ABBREVIATION"]: t.get("CONFERENCE", "?") for t in payload["teams"]}
    seed_of = {t["TEAM_ABBREVIATION"]: t.get("SEED", 99) for t in payload["teams"]}
    fr = payload["first_round"]
    # group the 4 first-round series per conference, ordered by top seed
    by_conf = {}
    for m in fr:
        c = conf_of.get(m["TEAM_A"], "?")
        by_conf.setdefault(c, []).append(m)
    conf_finals = []
    for conf, series in by_conf.items():
        series = sorted(series, key=lambda m: min(seed_of.get(m["TEAM_A"], 99), seed_of.get(m["TEAM_B"], 99)))
        leaves = [Node(round=1, team_a=m["TEAM_A"], team_b=m["TEAM_B"]) for m in series]
        while len(leaves) < 4:
            leaves.append(leaves[-1])
        r2a = Node(round=2, child_a=leaves[0], child_b=leaves[1])
        r2b = Node(round=2, child_a=leaves[2], child_b=leaves[3])
        conf_finals.append(Node(round=3, child_a=r2a, child_b=r2b))
    return Node(round=4, child_a=conf_finals[0], child_b=conf_finals[1])


def load_and_predict(json_path: str, n_sims: int = 5000) -> dict:
    payload = json.loads(Path(json_path).read_text())
    return predict_new_season(payload, n_sims=n_sims)
