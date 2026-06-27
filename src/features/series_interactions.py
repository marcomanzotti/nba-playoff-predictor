"""FASE V2-2 — Feature di INTERAZIONE DI STILE per il modello di serie (matchup).

PROBLEMA: il dataset-serie attuale usa solo DIFFERENZE feature-per-feature
(d_<col> = A - B). Cosi il modello vede "A tira piu' di B da 3", ma NON vede
"A vive di tiro da 3 E B difende benissimo il perimetro" -> non cattura il
'forte contro X, debole contro Y' che e' l'essenza dei playoff.

SOLUZIONE: termini di interazione che incrociano l'OFFESA di una squadra con la
DIFESA dell'altra (e lo stile con lo stile).

VINCOLO NON NEGOZIABILE (simmetria del repo): ogni termine deve essere
ANTISIMMETRICO -> scambiando A<->B cambia segno. Cosi:
  - il mirror in series_dataset.py resta valido (basta negare la colonna);
  - la predizione simmetrica (_predict_symmetric) resta coerente;
  - il label balance resta ~0.500.

Forma canonica usata: PRODOTTO INCROCIATO DISPARI
    x = f_off(A)*f_def(B) - f_off(B)*f_def(A)
scambiando A<->B  ->  f_off(B)*f_def(A) - f_off(A)*f_def(B) = -x   (antisimmetrico)

Per gli "scontri di stile" simmetrici (es. pace vs pace) si usa una forma che
resta dispari moltiplicando per la differenza:
    x = (A+B)/2 normalizzato * (A-B)   -> antisimmetrico perche' (A-B) lo e'.

Le feature di squadra arrivano come due Series (fa, fb) gia' allineate alle
stesse colonne. I valori sono standardizzati (z-score sul pool stagione) a monte
da chi chiama? No: qui usiamo i valori grezzi ma CENTRATI rispetto a una media di
riferimento passata dal chiamante, per evitare che il prodotto sia dominato dalla
scala assoluta (es. rating ~110). Vedi _center.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _val(s: pd.Series, col: str, center: dict) -> float:
    """Valore della feature col per una squadra, CENTRATO sulla media di
    riferimento (cosi il prodotto incrociato misura lo scarto dalla norma, non la
    scala assoluta). NaN -> 0 (= 'nella media')."""
    v = s.get(col, np.nan)
    if pd.isna(v):
        return 0.0
    return float(v) - center.get(col, 0.0)


def _cross(fa, fb, off_col, def_col, center) -> float:
    """Prodotto incrociato dispari: offesa di A vs difesa di B, meno il simmetrico.
    NB: per DEF_RATING piu' BASSO = difesa migliore, quindi a chi chiama spetta
    passare il segno giusto; qui trattiamo i valori gia' orientati."""
    a_off, a_def = _val(fa, off_col, center), _val(fa, def_col, center)
    b_off, b_def = _val(fb, off_col, center), _val(fb, def_col, center)
    return a_off * b_def - b_off * a_def


def _style_clash(fa, fb, col, center) -> float:
    """Scontro di stile antisimmetrico: chi e' piu' estremo nello stile, pesato
    dalla differenza. (media scarto) * (differenza) -> dispari in A<->B."""
    a, b = _val(fa, col, center), _val(fb, col, center)
    return ((a + b) / 2.0) * (a - b)


def interaction_terms(fa: pd.Series, fb: pd.Series, center: dict) -> dict:
    """Tutti i termini di interazione x_* per la coppia (A, B). Antisimmetrici.

    center: media di riferimento per colonna (es. media league di quella
    stagione), per centrare i valori prima dei prodotti.
    """
    out: dict[str, float] = {}

    # --- offesa vs difesa: chi attacca meglio CONTRO chi difende meglio ---
    # DEF_RATING: piu' basso = meglio -> lo invertiamo (neg) cosi "alto = difesa
    # forte" e il prodotto ha segno interpretabile.
    fa_d = fa.copy(); fb_d = fb.copy()
    if "DEF_RATING" in fa_d:
        fa_d["DEF_GOOD"] = -fa_d.get("DEF_RATING", np.nan)
        fb_d["DEF_GOOD"] = -fb_d.get("DEF_RATING", np.nan)
        center = {**center, "DEF_GOOD": -center.get("DEF_RATING", 0.0)}
    out["x_off_vs_def"] = _cross(fa_d, fb_d, "OFF_RATING", "DEF_GOOD", center)

    # --- tiro da 3 vs difesa: una squadra che tira tanto da 3 contro una buona
    #     difesa (proxy: DEF_GOOD) e' un matchup chiave dei playoff ---
    out["x_3pt_vs_def"] = _cross(fa_d, fb_d, "T4_n_shooters", "DEF_GOOD", center)

    # --- size mismatch nel frontcourt: lunghi di A vs lunghi di B ---
    out["x_fron_size"] = _style_clash(fa, fb, "BAND_fron_height", center)
    out["x_back_size"] = _style_clash(fa, fb, "BAND_back_height", center)

    # --- atletismo (vertical) come scontro di stile ---
    out["x_athleticism"] = _style_clash(fa, fb, "T2_vertical", center)

    # --- pace clash: ritmo contro ritmo (chi impone il proprio gioco) ---
    out["x_pace_clash"] = _style_clash(fa, fb, "PACE", center)

    # --- spacing vs rim: tiro perimetrale di A contro size interna di B ---
    a3 = _val(fa, "T4_n_shooters", center); b_size = _val(fb, "BAND_fron_height", center)
    b3 = _val(fb, "T4_n_shooters", center); a_size = _val(fa, "BAND_fron_height", center)
    out["x_spacing_vs_size"] = a3 * b_size - b3 * a_size

    return out


# colonne di squadra usate qui (per calcolare il center una volta sola a monte)
USED_COLS = ["OFF_RATING", "DEF_RATING", "T4_n_shooters",
             "BAND_fron_height", "BAND_back_height", "T2_vertical", "PACE"]


def league_center(tf_season: pd.DataFrame) -> dict:
    """Media di riferimento per colonna su una stagione (per centrare i prodotti)."""
    return {c: float(tf_season[c].mean()) for c in USED_COLS if c in tf_season.columns}
