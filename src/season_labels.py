"""Etichette di stagione — UNICA fonte di verita' (richiesta esplicita utente).

Regola FONDAMENTALE: una stagione si identifica con la notazione 'YYYY-YY' e il
suo CAMPIONE e' il vincitore del titolo nell'anno SOLARE FINALE.

  SEASON_START_YEAR = 2021  <->  SEASON = '2021-22'  <->  CHAMPIONS = 2022 (GSW)
  SEASON_START_YEAR = 2020  <->  SEASON = '2020-21'  <->  CHAMPIONS = 2021 (MIL)

Non usare MAI l'anno singolo per riferirsi al campione: e' ambiguo. Usare
season_label() o title_year() ovunque (report, viz, log).
"""
from __future__ import annotations


def season_label(start_year: int) -> str:
    """2021 -> '2021-22' (la stagione)."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def title_year(start_year: int) -> int:
    """2021 -> 2022 (l'anno solare in cui si assegna il titolo di quella stagione)."""
    return start_year + 1


def label_with_title(start_year: int) -> str:
    """2021 -> '2021-22 (titolo 2022)' — etichetta esplicita per le viz."""
    return f"{season_label(start_year)} (titolo {title_year(start_year)})"
