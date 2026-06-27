"""Layer di caching su disco + client nba_api robusto.

Tutte le chiamate alle API passano da qui, cosi:
  - ogni risultato e' salvato su disco (parquet) e non si ri-scarica;
  - rispettiamo il rate-limit di stats.nba.com (pausa tra chiamate);
  - i timeout/errori transitori vengono ritentati con backoff.

Uso tipico (dentro un collector):

    from src.collect.cache import cached_endpoint
    df = cached_endpoint(
        key="player_base/1996-97",
        endpoint_cls=leaguedashplayerstats.LeagueDashPlayerStats,
        params=dict(season="1996-97", measure_type_detailed_defense="Base"),
        frame_index=0,
    )
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

# Parametri rate-limit / retry (allineati a config.yaml::collection).
REQUEST_SLEEP_SEC = 0.6
MAX_RETRIES = 3
TIMEOUT_SEC = 60

# Timestamp dell'ultima richiesta di rete, per throttling globale.
_last_request_ts = 0.0


def _throttle() -> None:
    """Garantisce almeno REQUEST_SLEEP_SEC tra due chiamate di rete reali."""
    global _last_request_ts
    elapsed = time.time() - _last_request_ts
    if elapsed < REQUEST_SLEEP_SEC:
        time.sleep(REQUEST_SLEEP_SEC - elapsed)
    _last_request_ts = time.time()


def _cache_path(key: str) -> Path:
    """key 'player_base/1996-97' -> data/raw/player_base/1996-97.parquet"""
    return RAW_DIR / f"{key}.parquet"


def cached_endpoint(
    key: str,
    endpoint_cls,
    params: dict,
    frame_index: int = 0,
    force: bool = False,
) -> pd.DataFrame:
    """Ritorna un DataFrame da un endpoint nba_api, con cache su disco.

    Args:
        key: identificatore univoco -> path del file cache (senza estensione).
        endpoint_cls: classe endpoint di nba_api (es. LeagueDashPlayerStats).
        params: kwargs da passare al costruttore dell'endpoint.
        frame_index: quale data_frame restituire (di solito 0).
        force: se True ignora la cache e ri-scarica.

    Ritorna un DataFrame. Se l'endpoint da' una tabella vuota la cache la
    registra comunque (DataFrame vuoto) per non ri-tentare all'infinito.
    """
    path = _cache_path(key)
    if path.exists() and not force:
        return pd.read_parquet(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _throttle()
            endpoint = endpoint_cls(timeout=TIMEOUT_SEC, **params)
            df = endpoint.get_data_frames()[frame_index]
            # pyarrow vuole tipi coerenti: tutto-NA -> object da' fastidio.
            df.to_parquet(path, index=False)
            return df
        except Exception as e:  # noqa: BLE001 - retry su qualsiasi errore transitorio
            last_err = e
            wait = REQUEST_SLEEP_SEC * (2 ** attempt)  # backoff esponenziale
            print(f"    [retry {attempt}/{MAX_RETRIES}] {key}: {type(e).__name__} -> attendo {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError(f"cached_endpoint fallito per '{key}' dopo {MAX_RETRIES} tentativi: {last_err}")


def is_cached(key: str) -> bool:
    return _cache_path(key).exists()


def load_cached(key: str) -> pd.DataFrame:
    return pd.read_parquet(_cache_path(key))


def season_str(start_year: int) -> str:
    """1996 -> '1996-97' (formato stagione NBA)."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"
