"""Team visual metadata: ESPN logo URLs + official-ish team colors.

Used by the dashboard for a sporty, modern look. Colors are public facts; logos
are loaded from ESPN's public CDN by abbreviation.
"""
from __future__ import annotations

# Map our (sometimes historical) abbreviations to ESPN logo slugs.
ESPN_SLUG = {
    "ATL": "atl", "BKN": "bkn", "NJN": "bkn", "BOS": "bos", "CHA": "cha",
    "CHH": "cha", "CHI": "chi", "CLE": "cle", "DAL": "dal", "DEN": "den",
    "DET": "det", "GSW": "gs", "HOU": "hou", "IND": "ind", "LAC": "lac",
    "LAL": "lal", "MEM": "mem", "MIA": "mia", "MIL": "mil", "MIN": "min",
    "NOP": "no", "NOH": "no", "NYK": "ny", "OKC": "okc", "SEA": "okc",
    "ORL": "orl", "PHI": "phi", "PHX": "phx", "POR": "por", "SAC": "sac",
    "SAS": "sa", "TOR": "tor", "UTA": "utah", "WAS": "wsh",
}

# Primary team color (public). Historical sigle reuse the franchise color.
TEAM_COLOR = {
    "ATL": "#E03A3E", "BKN": "#000000", "NJN": "#000000", "BOS": "#007A33",
    "CHA": "#1D1160", "CHH": "#00788C", "CHI": "#CE1141", "CLE": "#860038",
    "DAL": "#00538C", "DEN": "#0E2240", "DET": "#C8102E", "GSW": "#1D428A",
    "HOU": "#CE1141", "IND": "#002D62", "LAC": "#C8102E", "LAL": "#552583",
    "MEM": "#5D76A9", "MIA": "#98002E", "MIL": "#00471B", "MIN": "#0C2340",
    "NOP": "#0C2340", "NOH": "#0C2340", "NYK": "#F58426", "OKC": "#007AC1",
    "SEA": "#00653A", "ORL": "#0077C0", "PHI": "#006BB6", "PHX": "#1D1160",
    "POR": "#E03A3E", "SAC": "#5A2D81", "SAS": "#C4CED4", "TOR": "#CE1141",
    "UTA": "#002B5C", "WAS": "#002B5C",
}


def logo_url(abbr: str, size: int = 500) -> str:
    slug = ESPN_SLUG.get(abbr, abbr.lower())
    return f"https://a.espncdn.com/i/teamlogos/nba/{size}/{slug}.png"


def color(abbr: str) -> str:
    return TEAM_COLOR.get(abbr, "#FF6B35")
