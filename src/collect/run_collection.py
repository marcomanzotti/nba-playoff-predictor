"""Orchestratore Fase 1 — raccolta dati completa con caching.

Scarica tutto cio' che serve nell'orizzonte 1996-2025:
  - anagrafica + storia draft + Combine (2000-2025)
  - giocatore-stagione (base/adv x RS/PO x pergame/totals)
  - squadra-stagione + partite RS (head-to-head) + partite playoff (serie)

Tutto passa dal layer di cache: rilanciare e' idempotente (riprende dove era
rimasto, non ri-scarica). Pensato per girare in background.

Uso:
    python -m src.collect.run_collection --start 1996 --end 2025
    python -m src.collect.run_collection --start 2022 --end 2023   # test
"""
from __future__ import annotations

import argparse
import time

from src.collect.players import collect_all_player_seasons
from src.collect.players_meta import (
    collect_all_combine,
    collect_draft_history,
    collect_players_index,
)
from src.collect.teams import collect_all_team_seasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1996)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--combine-last", type=int, default=2025)
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 60)
    print(f"FASE 1 — RACCOLTA DATI  ({args.start} .. {args.end})")
    print("=" * 60)

    print("\n[META] anagrafica + draft + combine")
    collect_players_index()
    collect_draft_history()
    # La Combine parte comunque dal 2000 a prescindere da --start.
    collect_all_combine(first=2000, last=args.combine_last)

    print("\n[TEAMS] stat squadra + partite RS/PO")
    collect_all_team_seasons(args.start, args.end)

    print("\n[PLAYERS] stat giocatore-stagione")
    collect_all_player_seasons(args.start, args.end)

    dt = time.time() - t0
    print("\n" + "=" * 60)
    print(f"FASE 1 COMPLETATA in {dt/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
