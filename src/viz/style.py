"""Shared visual style — modern, sporty, dark theme for all figures."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Palette: dark court background, vivid accents (NBA-ish)
BG = "#0E1117"
PANEL = "#161B22"
INK = "#E6EDF3"
MUTED = "#8B949E"
ACCENT = "#FF6B35"      # orange (basketball)
ACCENT2 = "#1D9BF0"     # blue
GOOD = "#2ECC71"
BAD = "#E74C3C"
GRID = "#2A2F37"

PALETTE = ["#FF6B35", "#1D9BF0", "#2ECC71", "#F1C40F", "#9B59B6", "#E74C3C", "#1ABC9C"]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": PANEL,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "legend.frameon": False,
    })


def savefig(fig, path, dpi: int = 130) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
