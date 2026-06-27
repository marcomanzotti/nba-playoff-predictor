"""Ricostruzione delle SERIE playoff e del record HEAD-TO-HEAD di RS.

Le serie sono il target del modello di singola serie (Fase 3); l'head-to-head
RS e' una feature di matchup centrale (la tesi: "SF batte OKC ma perde con NY").

Da `games_po/{season}.parquet` (righe-partita) ricostruiamo, per ogni coppia
di squadre che si sono incontrate ai playoff:
  - vittorie di ciascuna, vincitore della serie, numero gare;
  - il ROUND, dedotto dall'ordine cronologico (1=primo turno ... 4=Finals).

Da `games_rs/{season}.parquet` costruiamo la matrice head-to-head RS.

Output (data/interim/):
  playoff_series.parquet      una riga per serie-stagione
  h2h_records.parquet         una riga per (stagione, team, opp)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)


def _opponent_from_matchup(matchup: pd.Series) -> pd.Series:
    """'BOS vs. DAL' o 'BOS @ DAL' -> 'DAL'."""
    return matchup.str.extract(r"(?:vs\.|@)\s+(\w+)", expand=False)


def _ensure_season_cols(g: pd.DataFrame, season: str, start_year: int) -> pd.DataFrame:
    """I file raw in cache sono grezzi (senza SEASON). Le iniettiamo qui dal
    contesto (nome file), cosi la ricostruzione non dipende dal raw."""
    if "SEASON" not in g.columns:
        g["SEASON"] = season
    if "SEASON_START_YEAR" not in g.columns:
        g["SEASON_START_YEAR"] = start_year
    return g


def reconstruct_series_for_season(
    games_po: pd.DataFrame, season: str | None = None, start_year: int | None = None
) -> pd.DataFrame:
    """Ricostruisce le serie di una stagione dalle righe-partita playoff."""
    if games_po.empty:
        return pd.DataFrame()

    g = games_po.copy()
    if season is not None:
        g = _ensure_season_cols(g, season, start_year)
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    g["TEAM"] = g["TEAM_ABBREVIATION"]
    g["OPP"] = _opponent_from_matchup(g["MATCHUP"])

    # Chiave di serie indipendente dall'ordine (coppia non ordinata).
    g["PAIR"] = g.apply(lambda r: "__".join(sorted([r["TEAM"], r["OPP"]])), axis=1)

    rows = []
    for pair, sub in g.groupby("PAIR"):
        teams = pair.split("__")
        a, b = teams[0], teams[1]
        a_rows = sub[sub["TEAM"] == a]
        wins_a = int((a_rows["WL"] == "W").sum())
        wins_b = int((a_rows["WL"] == "L").sum())  # ogni gara di a e' una gara di b
        n_games = wins_a + wins_b
        winner = a if wins_a > wins_b else b
        loser = b if winner == a else a
        rows.append({
            "SEASON": sub["SEASON"].iloc[0],
            "SEASON_START_YEAR": int(sub["SEASON_START_YEAR"].iloc[0]),
            "TEAM_A": a, "TEAM_B": b,
            "WINS_A": wins_a, "WINS_B": wins_b,
            "N_GAMES": n_games,
            "WINNER": winner, "LOSER": loser,
            "SERIES_START": sub["GAME_DATE"].min(),
            "SERIES_END": sub["GAME_DATE"].max(),
        })

    series = pd.DataFrame(rows).sort_values("SERIES_START").reset_index(drop=True)
    series = _infer_rounds(series)
    return series


def _infer_rounds(series: pd.DataFrame) -> pd.DataFrame:
    """Deduce il round playoff dall'ordine cronologico di inizio serie.

    Logica robusta basata sul numero di serie attese per round nell'era a 16
    squadre (post-1984): 8 serie 1o turno, 4 semifinali conf, 2 finali conf,
    1 Finals. Usiamo il rank temporale di inizio serie.
    """
    s = series.copy()
    n = len(s)
    # Mappa standard del bracket a 16 squadre: 8 + 4 + 2 + 1 = 15 serie.
    # Assegniamo i round in base all'ordine di inizio.
    round_sizes = [(1, 8), (2, 4), (3, 2), (4, 1)]
    labels = []
    for rnd, size in round_sizes:
        labels += [rnd] * size
    # Se il numero di serie non e' 15 (es. play-in moderno non incluso, o dati
    # parziali), tronchiamo/estendiamo in modo difensivo.
    if n <= len(labels):
        s["ROUND"] = labels[:n]
    else:
        s["ROUND"] = labels + [4] * (n - len(labels))
    round_names = {1: "First Round", 2: "Conf Semifinals", 3: "Conf Finals", 4: "NBA Finals"}
    s["ROUND_NAME"] = s["ROUND"].map(round_names)
    return s


def build_all_series(start: int, end: int) -> pd.DataFrame:
    frames = []
    for yr in range(start, end + 1):
        season = f"{yr}-{str(yr+1)[-2:]}"
        p = RAW / "games_po" / f"{season}.parquet"
        if not p.exists():
            continue
        df = reconstruct_series_for_season(pd.read_parquet(p), season=season, start_year=yr)
        if not df.empty:
            frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_parquet(INTERIM / "playoff_series.parquet", index=False)
    print(f"playoff_series: {len(out)} serie su {out['SEASON'].nunique() if len(out) else 0} stagioni")
    return out


def build_h2h_for_season(
    games_rs: pd.DataFrame, season: str | None = None, start_year: int | None = None
) -> pd.DataFrame:
    """Record head-to-head di RS: per ogni (team, opp) vittorie/sconfitte."""
    g = games_rs.copy()
    if season is not None:
        g = _ensure_season_cols(g, season, start_year)
    g["TEAM"] = g["TEAM_ABBREVIATION"]
    g["OPP"] = _opponent_from_matchup(g["MATCHUP"])
    g["WIN"] = (g["WL"] == "W").astype(int)
    rec = (
        g.groupby(["SEASON", "SEASON_START_YEAR", "TEAM", "OPP"])
        .agg(GAMES=("WIN", "size"), WINS=("WIN", "sum"))
        .reset_index()
    )
    rec["LOSSES"] = rec["GAMES"] - rec["WINS"]
    rec["WIN_PCT"] = rec["WINS"] / rec["GAMES"]
    return rec


def build_all_h2h(start: int, end: int) -> pd.DataFrame:
    frames = []
    for yr in range(start, end + 1):
        season = f"{yr}-{str(yr+1)[-2:]}"
        p = RAW / "games_rs" / f"{season}.parquet"
        if not p.exists():
            continue
        frames.append(build_h2h_for_season(pd.read_parquet(p), season=season, start_year=yr))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out.to_parquet(INTERIM / "h2h_records.parquet", index=False)
    print(f"h2h_records: {len(out)} righe (team-opp) su {out['SEASON'].nunique() if len(out) else 0} stagioni")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1996)
    ap.add_argument("--end", type=int, default=2025)
    args = ap.parse_args()
    build_all_series(args.start, args.end)
    build_all_h2h(args.start, args.end)
