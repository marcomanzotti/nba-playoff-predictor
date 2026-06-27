"""Generate all key figures for the project (English, sporty dark theme).

Figures produced in reports/figures/:
  1. thesis_ablation.png        — does the thesis add value? (baseline/full/thesis-only)
  2. shap_factors.png           — what predicts who WINS a series (by factor)
  3. rating_drivers.png         — what BUILDS a good net rating (the causal chain)
  4. causal_chain.png           — players -> net rating -> playoff wins (diagram)
  5. champion_probabilities.png — predicted title odds vs real champion (backtest)
  6. calibration.png            — model calibration on the held-out test
  7. matchup_example.png        — one season's first-round matchup probabilities
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.viz.style import (ACCENT, ACCENT2, BAD, GOOD, INK, MUTED, PALETTE,
                           apply_style, savefig)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def fig_thesis_ablation() -> None:
    r = json.loads((PROCESSED / "thesis_test_results.json").read_text())
    models = [("A) Baseline\n(record, rating,\nhome court)", "baseline"),
              ("B) Full\n(+ thesis T1-T5)", "complete"),
              ("C) Thesis only\n(T1-T5, no record)", "thesis_only")]
    acc = [r[k]["test"]["accuracy"] for _, k in models]
    ll = [r[k]["test"]["log_loss"] for _, k in models]
    labels = [m for m, _ in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    bars = ax1.bar(labels, acc, color=[MUTED, ACCENT, ACCENT2])
    ax1.axhline(0.5, color=BAD, ls="--", lw=1, label="coin flip")
    ax1.set_ylim(0.4, 0.8)
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("Who wins a series? — accuracy")
    for b, v in zip(bars, acc):
        ax1.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.0%}", ha="center", color=INK, fontweight="bold")
    ax1.legend()

    bars2 = ax2.bar(labels, ll, color=[MUTED, ACCENT, ACCENT2])
    ax2.set_ylabel("Log loss (lower = better)")
    ax2.set_title("Probability quality — log loss")
    for b, v in zip(bars2, ll):
        ax2.text(b.get_x()+b.get_width()/2, v+0.004, f"{v:.3f}", ha="center", color=INK, fontweight="bold")

    fig.suptitle("Does the thesis add predictive value to WINNING a series?")
    savefig(fig, FIG / "thesis_ablation.png")


def fig_shap_factors() -> None:
    s = json.loads((PROCESSED / "shap_summary.json").read_text())
    bf = s["by_factor"]
    names = {"HOME_COURT": "Home court", "ROLE": "Role quality", "TEAM_STATS": "Team rating",
             "T3": "T3 Playoff exp", "T5": "T5 Involvement", "T2": "T2 Size/athletic",
             "T1": "T1 Homegrown", "T4": "T4 Three-point", "ROSTER_QUALITY": "Roster (stars)",
             "MATCHUP_H2H": "Head-to-head"}
    items = sorted(bf.items(), key=lambda kv: kv[1])
    labels = [names.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    colors = [ACCENT if k.startswith("T") and k != "TEAM_STATS" else MUTED for k, _ in items]

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Mean |SHAP| (impact on series outcome)")
    ax.set_title("What predicts WHO WINS a series\n(orange = thesis factors)")
    savefig(fig, FIG / "shap_factors.png")


def fig_rating_drivers() -> None:
    r = json.loads((PROCESSED / "rating_drivers.json").read_text())
    bf = r["by_factor"]
    names = {"ROLE": "Role quality\n(strong at every position)", "T5": "T5 Involvement",
             "T4": "T4 Three-point shooting", "ROSTER_QUALITY": "Star power",
             "T3": "T3 Playoff experience", "T2": "T2 Size / athleticism",
             "T1": "T1 Homegrown", "OTHER": "Other"}
    items = sorted(bf.items(), key=lambda kv: kv[1])
    labels = [names.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    colors = [ACCENT if k.startswith("T") else (ACCENT2 if k == "ROLE" else MUTED) for k, _ in items]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh(labels, vals, color=colors)
    ax.set_xlabel("Contribution to NET RATING (mean |SHAP|)")
    ax.set_title(f"What BUILDS a strong team (net rating)\n"
                 f"player features explain R²={r['r2_test']:.0%} of net rating")
    savefig(fig, FIG / "rating_drivers.png")


def fig_causal_chain() -> None:
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.axis("off")
    boxes = [
        (0.13, "PLAYER FEATURES\nT2 size · T3 experience\nT4 shooting · role quality", ACCENT),
        (0.5, "HIGH NET RATING\n(the strong team)", ACCENT2),
        (0.87, "PLAYOFF WINS\n(+ home court)", GOOD),
    ]
    for x, txt, col in boxes:
        ax.add_patch(plt.Rectangle((x-0.14, 0.3), 0.28, 0.4, transform=ax.transAxes,
                     facecolor=col, alpha=0.18, edgecolor=col, lw=2))
        ax.text(x, 0.5, txt, transform=ax.transAxes, ha="center", va="center",
                color=INK, fontweight="bold", fontsize=11)
    for x0, x1 in [(0.27, 0.36), (0.64, 0.73)]:
        ax.annotate("", xy=(x1, 0.5), xytext=(x0, 0.5), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.5))
    ax.text(0.5, 0.92, "How a team wins: the thesis acts UPSTREAM, through net rating",
            transform=ax.transAxes, ha="center", color=INK, fontsize=13, fontweight="bold")
    savefig(fig, FIG / "causal_chain.png")


def fig_champion_probabilities() -> None:
    summ = json.loads((PROCESSED / "bracket_backtest_summary.json").read_text())
    rows = []
    for yr, d in summ.items():
        rows.append({"season": d.get("season", yr), "champ": d["champion_real"],
                     "p": d["prob_assigned_to_real_champion"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(df["season"], df["p"], color=ACCENT)
    ax.axhline(1/16, color=MUTED, ls="--", lw=1, label="random (1/16)")
    ax.set_ylabel("Model's title probability for the\nactual champion (pre-playoffs)")
    ax.set_title("Backtest: did the model see the real champion coming?")
    for b, (_, row) in zip(bars, df.iterrows()):
        ax.text(b.get_x()+b.get_width()/2, row["p"]+0.01, row["champ"],
                ha="center", color=INK, fontweight="bold", fontsize=9)
    ax.legend()
    savefig(fig, FIG / "champion_probabilities.png")


def fig_calibration() -> None:
    test = pd.read_parquet(PROCESSED / "walkforward_test_preds.parquet")
    bins = np.linspace(0, 1, 6)
    test["bin"] = pd.cut(test["p"], bins=bins)
    cal = test.groupby("bin", observed=True).agg(pred=("p", "mean"), real=("y", "mean"))
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", label="perfect")
    ax.plot(cal["pred"], cal["real"], "o-", color=ACCENT, lw=2, ms=8, label="model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed win rate")
    ax.set_title("Model calibration (held-out test)")
    ax.legend()
    savefig(fig, FIG / "calibration.png")


def fig_matchup_example(year: int = 2023) -> None:
    p = PROCESSED / f"season_report_{year}_matchups.parquet"
    if not p.exists():
        return
    mu = pd.read_parquet(p)
    m = mu[mu["ROUND"] == 1].copy()
    m["pair"] = m["TEAM_A"] + " vs " + m["TEAM_B"]
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(m))
    ax.barh(y, m["P_A_wins_series"], color=ACCENT, label="Team A")
    ax.barh(y, m["P_B_wins_series"], left=m["P_A_wins_series"], color=ACCENT2, label="Team B")
    ax.set_yticks(y)
    ax.set_yticklabels(m["pair"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Series win probability")
    ax.set_title(f"First-round series probabilities — {m['SEASON'].iloc[0]}")
    for i, row in enumerate(m.itertuples()):
        ax.text(row.P_A_wins_series/2, i, f"{row.P_A_wins_series:.0%}", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax.legend(loc="lower right")
    savefig(fig, FIG / "matchup_example.png")


def main() -> None:
    apply_style()
    fig_thesis_ablation()
    fig_shap_factors()
    fig_rating_drivers()
    fig_causal_chain()
    fig_champion_probabilities()
    fig_calibration()
    fig_matchup_example()
    print(f"Figures written to {FIG.relative_to(ROOT)}/")
    for f in sorted(FIG.glob("*.png")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
