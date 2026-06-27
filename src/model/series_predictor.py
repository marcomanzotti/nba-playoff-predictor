"""Predittore P(A batte B) per QUALSIASI coppia in una stagione (Fase 4).

Riusa la stessa logica di feature differenziali di series_dataset, ma esposta
come funzione chiamabile su due squadre arbitrarie (servira' al simulatore di
bracket, che deve valutare accoppiamenti mai avvenuti nella realta').

Carica una volta team_features e h2h; espone:
  predictor = SeriesPredictor(model)
  p = predictor.prob(year, "BOS", "MIA")   # P(BOS vince la serie)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"

_DROP = {"SEASON_START_YEAR", "TEAM_ID"}


def feature_columns(tf: pd.DataFrame) -> list[str]:
    num = tf.select_dtypes("number").columns.tolist()
    return [c for c in num if c not in _DROP]


class SeriesPredictor:
    def __init__(self, model, feat_order: list[str] | None = None):
        self.model = model
        self.tf = pd.read_parquet(PROCESSED / "team_season_features.parquet")
        self.h2h = pd.read_parquet(INTERIM / "h2h_records.parquet")
        self.conf = pd.read_parquet(INTERIM / "team_conference.parquet")
        self.team_cols = feature_columns(self.tf)
        self.tf_idx = self.tf.set_index(["SEASON_START_YEAR", "TEAM_ABBREVIATION"])
        self.h2h_idx = self.h2h.set_index(["SEASON_START_YEAR", "TEAM", "OPP"])
        self.seed_idx = self.conf.set_index(["SEASON_START_YEAR", "TEAM_ABBREVIATION"])["SEED"]
        # ordine colonne atteso dal modello: d_<col> + H2H_DIFF + HOME_COURT
        self.feat_order = feat_order or (
            [f"d_{c}" for c in self.team_cols] + ["H2H_DIFF", "HOME_COURT"]
        )
        # durante una simulazione le stesse coppie si ripetono migliaia di volte
        self._cache: dict[tuple, float] = {}

    def _team_vec(self, year: int, team: str) -> pd.Series | None:
        try:
            row = self.tf_idx.loc[(year, team), self.team_cols]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row

    def _h2h_wins(self, year, team, opp) -> float:
        try:
            return float(self.h2h_idx.loc[(year, team, opp), "WINS"])
        except KeyError:
            return 0.0

    def _home_court(self, year, a, b) -> float:
        try:
            sa = float(self.seed_idx.loc[(year, a)])
            sb = float(self.seed_idx.loc[(year, b)])
        except KeyError:
            return 0.0
        return 1.0 if sa < sb else (-1.0 if sb < sa else 0.0)

    def diff_vector(self, year: int, a: str, b: str) -> pd.DataFrame | None:
        fa = self._team_vec(year, a)
        fb = self._team_vec(year, b)
        if fa is None or fb is None:
            return None
        diff = (fa - fb).astype(float)
        h2h_diff = self._h2h_wins(year, a, b) - self._h2h_wins(year, b, a)
        row = {f"d_{c}": diff[c] for c in self.team_cols}
        row["H2H_DIFF"] = h2h_diff
        row["HOME_COURT"] = self._home_court(year, a, b)
        x = pd.DataFrame([row])[self.feat_order].fillna(0.0)
        return x

    def prob(self, year: int, a: str, b: str) -> float:
        """P(a vince la serie), simmetrica: media vista diretta e speculare.
        Memoizzata per (year,a,b) -> grande speedup nel Monte Carlo."""
        key = (year, a, b)
        if key in self._cache:
            return self._cache[key]
        x = self.diff_vector(year, a, b)
        if x is None:
            return 0.5
        xv = x.to_numpy()
        p_direct = self.model.predict_proba(xv)[0, 1]
        p_mirror = self.model.predict_proba(-xv)[0, 1]
        p = float((p_direct + (1 - p_mirror)) / 2)
        self._cache[key] = p
        self._cache[(year, b, a)] = 1 - p  # simmetria
        return p
