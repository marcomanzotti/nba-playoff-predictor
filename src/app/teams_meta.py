"""Team visual metadata: official NBA logos + trophy + team colors.

Used by the dashboard for a sporty, modern look with the *real* NBA imagery.
Everything here is loaded from public CDNs (NBA's own CDN, ESPN's CDN, and
Wikimedia Commons). This is a public, non-commercial educational repo.

Logo strategy:
  - Primary: NBA's official CDN by stable franchise TEAM_ID (high-res SVG/PNG).
  - Fallback: ESPN's CDN by abbreviation (used in the dashboard <img onerror>).
Historical abbreviations (CHH, NJN, SEA, NOH...) reuse the current franchise id.
"""
from __future__ import annotations

# Stable NBA franchise id per (sometimes historical) abbreviation. Relocations
# keep the same franchise id, so we map old sigle to the current franchise.
TEAM_ID = {
    "ATL": 1610612737, "BKN": 1610612751, "NJN": 1610612751,
    "BOS": 1610612738, "CHA": 1610612766, "CHH": 1610612766,
    "CHI": 1610612741, "CLE": 1610612739, "DAL": 1610612742,
    "DEN": 1610612743, "DET": 1610612765, "GSW": 1610612744,
    "HOU": 1610612745, "IND": 1610612754, "LAC": 1610612746,
    "LAL": 1610612747, "MEM": 1610612763, "VAN": 1610612763,
    "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
    "NOP": 1610612740, "NOH": 1610612740, "NOK": 1610612740,
    "NYK": 1610612752, "OKC": 1610612760, "SEA": 1610612760,
    "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
    "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
    "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
}

# Map our (sometimes historical) abbreviations to ESPN logo slugs (fallback).
ESPN_SLUG = {
    "ATL": "atl", "BKN": "bkn", "NJN": "bkn", "BOS": "bos", "CHA": "cha",
    "CHH": "cha", "CHI": "chi", "CLE": "cle", "DAL": "dal", "DEN": "den",
    "DET": "det", "GSW": "gs", "HOU": "hou", "IND": "ind", "LAC": "lac",
    "LAL": "lal", "MEM": "mem", "VAN": "mem", "MIA": "mia", "MIL": "mil",
    "MIN": "min", "NOP": "no", "NOH": "no", "NOK": "no", "NYK": "ny",
    "OKC": "okc", "SEA": "okc", "ORL": "orl", "PHI": "phi", "PHX": "phx",
    "POR": "por", "SAC": "sac", "SAS": "sa", "TOR": "tor", "UTA": "utah",
    "WAS": "wsh",
}

# Primary team color (public fact). Historical sigle reuse the franchise color.
TEAM_COLOR = {
    "ATL": "#E03A3E", "BKN": "#000000", "NJN": "#000000", "BOS": "#007A33",
    "CHA": "#1D1160", "CHH": "#00788C", "CHI": "#CE1141", "CLE": "#860038",
    "DAL": "#00538C", "DEN": "#0E2240", "DET": "#C8102E", "GSW": "#1D428A",
    "HOU": "#CE1141", "IND": "#002D62", "LAC": "#C8102E", "LAL": "#552583",
    "MEM": "#5D76A9", "VAN": "#5D76A9", "MIA": "#98002E", "MIL": "#00471B",
    "MIN": "#0C2340", "NOP": "#0C2340", "NOH": "#0C2340", "NOK": "#0C2340",
    "NYK": "#F58426", "OKC": "#007AC1", "SEA": "#00653A", "ORL": "#0077C0",
    "PHI": "#006BB6", "PHX": "#1D1160", "POR": "#E03A3E", "SAC": "#5A2D81",
    "SAS": "#C4CED4", "TOR": "#CE1141", "UTA": "#002B5C", "WAS": "#002B5C",
}

# Real Larry O'Brien Championship Trophy (Wikimedia Commons, public).
TROPHY_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/"
    "Larry_O%27Brien_Championship_Trophy_icon.svg/"
    "330px-Larry_O%27Brien_Championship_Trophy_icon.svg.png"
)
# Official NBA league logo (PNG; renders reliably cross-origin in the browser).
NBA_LOGO_URL = "https://a.espncdn.com/i/teamlogos/leagues/500/nba.png"


def logo_url(abbr: str, size: int = 500) -> str:
    """Real NBA team logo for the dashboard.

    We serve the ESPN-CDN PNG: it's the *real* NBA team logo and, unlike the NBA
    CDN's SVGs, it renders reliably as a cross-origin <img> in the browser (the
    NBA-CDN SVGs come back 0x0 due to hotlink/CORS handling). The official
    NBA-CDN URL is available via `nba_cdn_logo_url()` for docs/reference.
    """
    return espn_logo_url(abbr, size)


def espn_logo_url(abbr: str, size: int = 500) -> str:
    """ESPN-CDN logo by abbreviation (the real team logo, browser-friendly)."""
    slug = ESPN_SLUG.get(abbr, abbr.lower())
    return f"https://a.espncdn.com/i/teamlogos/nba/{size}/{slug}.png"


def nba_cdn_logo_url(abbr: str) -> str | None:
    """Official NBA-CDN logo URL by franchise id (high-res SVG; reference only)."""
    tid = TEAM_ID.get(abbr)
    return f"https://cdn.nba.com/logos/nba/{tid}/primary/L/logo.svg" if tid else None


def color(abbr: str) -> str:
    return TEAM_COLOR.get(abbr, "#FF6B35")
