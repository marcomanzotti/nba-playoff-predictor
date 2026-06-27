"""Fase 0 — Audit copertura dati.

Verifica, PRIMA di costruire la pipeline, che le fonti dati coprano davvero
quello che serve sulle 30 stagioni (1996-97 .. 2025-26):

  1. Connettivita e rate-limit di nba_api.
  2. Dati Combine (anthropometrics + atletismo): quanti giocatori per anno,
     da che anno partono. Critico per il clustering proxy (spec §3.5).
  3. Stat per giocatore-stagione (box + advanced) su un campione di stagioni.
  4. Stat di squadra + record head-to-head.
  5. Tabellone playoff / serie (per il modello di singola serie).

Stampa un report leggibile e salva un JSON in data/interim/audit_report.json.

Uso:
    python -m src.collect.audit            # esegue tutto
    python -m src.collect.audit --quick    # campione ridotto, piu veloce
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "interim"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLEEP = 0.6  # pausa tra chiamate per rispettare il rate limit di stats.nba.com


def season_str(start_year: int) -> str:
    """1996 -> '1996-97' (formato stagione NBA)."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _try(label, fn):
    """Esegue fn() catturando l'errore; ritorna (ok, value_or_error)."""
    try:
        val = fn()
        return True, val
    except Exception as e:  # noqa: BLE001 - vogliamo vedere QUALSIASI errore
        return False, f"{type(e).__name__}: {e}"


def check_connectivity(report: dict) -> None:
    print("\n[1/5] Connettivita nba_api ...")
    from nba_api.stats.endpoints import leaguedashteamstats

    ok, val = _try(
        "team_stats_current",
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season_str(2024)
        ).get_data_frames()[0],
    )
    if ok:
        report["connectivity"] = {"ok": True, "n_teams": len(val), "cols": list(val.columns)[:12]}
        print(f"    OK — {len(val)} squadre ricevute per la stagione 2024-25.")
    else:
        report["connectivity"] = {"ok": False, "error": val}
        print(f"    FALLITO — {val}")


def check_combine(report: dict, seasons: list[int]) -> None:
    print("\n[2/5] Dati Combine (anthro + atletismo) ...")
    from nba_api.stats.endpoints import draftcombinestats

    rows = []
    for yr in seasons:
        # La Combine si indicizza con l'anno solare dell'evento (stringa),
        # parametro `season_all_time` (es. '2024').
        ok, df = _try(
            f"combine_{yr}",
            lambda yr=yr: draftcombinestats.DraftCombineStats(
                season_all_time=str(yr)
            ).get_data_frames()[0],
        )
        time.sleep(SLEEP)
        if not ok:
            rows.append({"season": yr, "ok": False, "error": df})
            print(f"    {season_str(yr)}: ERRORE {df}")
            continue
        # quante misure chiave sono effettivamente popolate?
        anthro = [c for c in ("HEIGHT_WO_SHOES", "WINGSPAN", "STANDING_REACH") if c in df.columns]
        athl = [c for c in ("MAX_VERTICAL_LEAP", "LANE_AGILITY_TIME", "THREE_QUARTER_SPRINT") if c in df.columns]
        n = len(df)
        anthro_filled = int(df[anthro].notna().any(axis=1).sum()) if anthro and n else 0
        athl_filled = int(df[athl].notna().any(axis=1).sum()) if athl and n else 0
        rows.append({
            "season": yr, "ok": True, "n_players": n,
            "with_anthro": anthro_filled, "with_athleticism": athl_filled,
        })
        print(f"    {season_str(yr)}: {n} giocatori | anthro={anthro_filled} | atletismo={athl_filled}")
    report["combine"] = rows


def check_player_season(report: dict, seasons: list[int]) -> None:
    print("\n[3/5] Stat per giocatore-stagione (box + advanced) ...")
    from nba_api.stats.endpoints import leaguedashplayerstats

    rows = []
    for yr in seasons:
        ok_base, base = _try(
            f"pstats_base_{yr}",
            lambda yr=yr: leaguedashplayerstats.LeagueDashPlayerStats(
                season=season_str(yr), measure_type_detailed_defense="Base"
            ).get_data_frames()[0],
        )
        time.sleep(SLEEP)
        ok_adv, adv = _try(
            f"pstats_adv_{yr}",
            lambda yr=yr: leaguedashplayerstats.LeagueDashPlayerStats(
                season=season_str(yr), measure_type_detailed_defense="Advanced"
            ).get_data_frames()[0],
        )
        time.sleep(SLEEP)
        entry = {
            "season": yr,
            "base_ok": ok_base, "n_players_base": len(base) if ok_base else None,
            "adv_ok": ok_adv, "n_players_adv": len(adv) if ok_adv else None,
        }
        if not ok_base:
            entry["base_error"] = base
        if not ok_adv:
            entry["adv_error"] = adv
        rows.append(entry)
        nb = len(base) if ok_base else "ERR"
        na = len(adv) if ok_adv else "ERR"
        print(f"    {season_str(yr)}: base={nb} | advanced={na}")
    report["player_season"] = rows


def check_team_and_h2h(report: dict, season: int) -> None:
    print(f"\n[4/5] Stat squadra + head-to-head ({season_str(season)}) ...")
    from nba_api.stats.endpoints import leaguedashteamstats, leaguegamefinder

    ok_team, team = _try(
        "team_stats",
        lambda: leaguedashteamstats.LeagueDashTeamStats(
            season=season_str(season), measure_type_detailed_defense="Advanced"
        ).get_data_frames()[0],
    )
    time.sleep(SLEEP)
    # head-to-head: tutte le partite di RS, da cui ricostruire i record reciproci
    ok_games, games = _try(
        "games",
        lambda: leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str(season),
            season_type_nullable="Regular Season",
            league_id_nullable="00",
        ).get_data_frames()[0],
    )
    time.sleep(SLEEP)
    report["team_h2h"] = {
        "season": season,
        "team_ok": ok_team, "n_teams": len(team) if ok_team else None,
        "team_error": None if ok_team else team,
        "games_ok": ok_games,
        "n_game_rows": len(games) if ok_games else None,
        "games_error": None if ok_games else games,
    }
    if ok_team:
        print(f"    squadre (advanced): {len(team)}")
    else:
        print(f"    squadre: ERRORE {team}")
    if ok_games:
        # ogni partita ha 2 righe (una per squadra) -> n_game_rows/2 partite
        print(f"    righe-partita RS: {len(games)} (~{len(games)//2} partite) -> head-to-head ricostruibile")
    else:
        print(f"    partite: ERRORE {games}")


def check_playoffs(report: dict, season: int) -> None:
    print(f"\n[5/5] Tabellone playoff / serie ({season_str(season)}) ...")
    from nba_api.stats.endpoints import leaguegamefinder

    ok, games = _try(
        "playoff_games",
        lambda: leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str(season),
            season_type_nullable="Playoffs",
            league_id_nullable="00",
        ).get_data_frames()[0],
    )
    time.sleep(SLEEP)
    report["playoffs"] = {
        "season": season, "ok": ok,
        "n_game_rows": len(games) if ok else None,
        "error": None if ok else games,
    }
    if ok:
        n_teams = games["TEAM_ID"].nunique() if "TEAM_ID" in games.columns else "?"
        print(f"    righe-partita playoff: {len(games)} (~{len(games)//2} partite) | squadre coinvolte: {n_teams}")
        print("    -> da MATCHUP/risultati gara-per-gara si ricostruiscono le serie.")
    else:
        print(f"    ERRORE {ok and '' or games}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="campione ridotto di stagioni")
    args = parser.parse_args()

    # Stagioni campione per l'audit (non scarichiamo tutto, solo verifichiamo
    # la copertura agli estremi e in mezzo dell'orizzonte 1996..2025).
    if args.quick:
        combine_seasons = [2000, 2012, 2024]
        pstats_seasons = [1996, 2010, 2024]
    else:
        combine_seasons = [1996, 2000, 2005, 2010, 2015, 2020, 2024]
        pstats_seasons = [1996, 2000, 2005, 2010, 2015, 2020, 2024]

    report = {"generated_at": datetime.now().isoformat(), "horizon": "1996-97 .. 2025-26"}

    print("=" * 64)
    print("FASE 0 — AUDIT COPERTURA DATI (NBA Playoff Predictor)")
    print("=" * 64)

    check_connectivity(report)
    # Se la connettivita fallisce del tutto, inutile proseguire.
    if not report["connectivity"].get("ok"):
        print("\n!! Connettivita fallita: salto il resto. Controlla rete/nba_api.")
    else:
        check_combine(report, combine_seasons)
        check_player_season(report, pstats_seasons)
        check_team_and_h2h(report, 2023)
        check_playoffs(report, 2023)

    out = OUT_DIR / "audit_report.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport salvato in {out.relative_to(ROOT)}")
    print("=" * 64)


if __name__ == "__main__":
    main()
