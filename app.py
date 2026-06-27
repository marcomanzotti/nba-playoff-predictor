"""NBA Playoff Predictor — interactive dashboard (sporty, modern).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.engine import (available_seasons, predict_season, real_outcome,
                            team_strengths)
from src.app.teams_meta import color as team_color
from src.app.teams_meta import logo_url
from src.season_labels import label_with_title, season_label

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="wide")

ORANGE, GOLD, BLUE, GREEN = "#FF6B35", "#FFC857", "#1D9BF0", "#2ECC71"
INK, MUTED, PANEL, BG = "#E6EDF3", "#8B949E", "#161B22", "#0B0E14"

st.markdown(f"""
<style>
  .stApp {{ background: radial-gradient(1200px 600px at 50% -10%, #1a2030 0%, {BG} 55%); }}
  #MainMenu, footer {{ visibility: hidden; }}
  h1,h2,h3,h4 {{ color:{INK}; font-weight:800; letter-spacing:.3px; }}
  .hero {{ text-align:center; padding: 1.6rem 1rem .6rem; }}
  .hero .trophy {{ font-size:3.2rem; filter: drop-shadow(0 0 18px {GOLD}); }}
  .hero h1 {{ font-size:2.6rem; margin:.2rem 0 0; background:linear-gradient(90deg,{GOLD},{ORANGE});
              -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .hero p {{ color:{MUTED}; margin:.2rem 0 0; font-size:1rem; }}
  .champ-card {{ background:linear-gradient(160deg,#20283a,{PANEL}); border:1px solid #2a3550;
                 border-radius:18px; padding:1.4rem; text-align:center;
                 box-shadow:0 8px 30px rgba(0,0,0,.4); }}
  .champ-card .lbl {{ color:{MUTED}; text-transform:uppercase; font-size:.75rem; letter-spacing:2px; }}
  .champ-card .team {{ font-size:2rem; font-weight:900; color:{INK}; }}
  .champ-card .pct {{ font-size:2.4rem; font-weight:900; color:{GOLD}; }}
  .prob-row {{ display:flex; align-items:center; gap:.6rem; padding:.35rem .5rem;
               border-radius:10px; margin-bottom:.25rem; background:{PANEL}; }}
  .prob-row img {{ width:30px; height:30px; }}
  .prob-row .abbr {{ width:46px; font-weight:800; color:{INK}; }}
  .bar-wrap {{ flex:1; background:#0d1117; border-radius:6px; height:20px; overflow:hidden; }}
  .bar {{ height:100%; border-radius:6px; }}
  .prob-row .val {{ width:52px; text-align:right; font-weight:800; color:{INK}; }}
  .matchup {{ display:flex; align-items:center; gap:1rem; background:{PANEL};
              border-radius:14px; padding:.7rem 1rem; margin-bottom:.5rem; }}
  .matchup img {{ width:42px; height:42px; }}
  .matchup .pct {{ font-weight:900; font-size:1.2rem; min-width:54px; text-align:center; }}
  .matchup .vs {{ color:{MUTED}; font-weight:700; }}
  .badge {{ display:inline-block; padding:.15rem .6rem; border-radius:20px;
            font-size:.78rem; font-weight:700; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <div class="trophy">🏆</div>
  <h1>NBA PLAYOFF PREDICTOR</h1>
  <p>Regular-season data → series-by-series playoff probabilities · walk-forward validated · 1996–2025</p>
</div>
""", unsafe_allow_html=True)


def _img(abbr, size=40):
    return f'<img src="{logo_url(abbr)}" width="{size}" onerror="this.style.opacity=0">'


def prob_rows(adv: pd.DataFrame) -> str:
    html = ""
    mx = max(adv["P_champion"].max(), 0.01)
    for _, r in adv.iterrows():
        pct = r["P_champion"]
        w = int(100 * pct / mx)
        col = team_color(r["TEAM"])
        html += (f'<div class="prob-row">{_img(r["TEAM"],30)}'
                 f'<span class="abbr">{r["TEAM"]}</span>'
                 f'<div class="bar-wrap"><div class="bar" style="width:{w}%;'
                 f'background:linear-gradient(90deg,{col},{GOLD})"></div></div>'
                 f'<span class="val">{pct:.0%}</span></div>')
    return html


def matchup_cards(mu: pd.DataFrame, rnd: int) -> str:
    html = ""
    for _, m in mu[mu["ROUND"] == rnd].iterrows():
        pa, pb = m["P_A_wins_series"], m["P_B_wins_series"]
        ca = GREEN if pa >= pb else MUTED
        cb = GREEN if pb > pa else MUTED
        html += (f'<div class="matchup">{_img(m["TEAM_A"],42)}'
                 f'<span class="pct" style="color:{ca}">{pa:.0%}</span>'
                 f'<span style="font-weight:800">{m["TEAM_A"]}</span>'
                 f'<span class="vs">vs</span>'
                 f'<span style="font-weight:800">{m["TEAM_B"]}</span>'
                 f'<span class="pct" style="color:{cb}">{pb:.0%}</span>'
                 f'{_img(m["TEAM_B"],42)}</div>')
    return html


def advancement_table(adv: pd.DataFrame) -> pd.DataFrame:
    d = adv.copy()
    for c in ["P_pass_R1", "P_pass_R2", "P_pass_ConfFinals", "P_champion"]:
        d[c] = (d[c] * 100).round(1)
    return d.rename(columns={"TEAM": "Team", "P_pass_R1": "Round 2 %",
                             "P_pass_R2": "Conf SF %", "P_pass_ConfFinals": "Conf Finals %",
                             "P_champion": "Champion %"})[
        ["Team", "Round 2 %", "Conf SF %", "Conf Finals %", "Champion %"]]


def render_prediction(adv, mu, real=None):
    champ = adv.iloc[0]
    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.markdown(
            f'<div class="champ-card"><div class="lbl">Predicted champion</div>'
            f'<img src="{logo_url(champ["TEAM"])}" width="86"><div class="team">{champ["TEAM"]}</div>'
            f'<div class="pct">{champ["P_champion"]:.0%}</div></div>', unsafe_allow_html=True)
    if real and real.get("champion"):
        with cols[1]:
            ok = champ["TEAM"] == real["champion"]
            st.markdown(
                f'<div class="champ-card"><div class="lbl">Actual champion</div>'
                f'<img src="{logo_url(real["champion"])}" width="86">'
                f'<div class="team">{real["champion"]}</div>'
                f'<div class="pct" style="color:{GREEN if ok else ORANGE}">'
                f'{"✅ HIT" if ok else "❌ MISS"}</div></div>', unsafe_allow_html=True)
        with cols[2]:
            runner = adv.iloc[1]
            st.markdown(
                f'<div class="champ-card"><div class="lbl">Top contender #2</div>'
                f'<img src="{logo_url(runner["TEAM"])}" width="86"><div class="team">{runner["TEAM"]}</div>'
                f'<div class="pct" style="color:{BLUE}">{runner["P_champion"]:.0%}</div></div>',
                unsafe_allow_html=True)
    else:
        for i, c in enumerate(cols[1:], start=1):
            r = adv.iloc[i]
            with c:
                st.markdown(
                    f'<div class="champ-card"><div class="lbl">Contender #{i+1}</div>'
                    f'<img src="{logo_url(r["TEAM"])}" width="86"><div class="team">{r["TEAM"]}</div>'
                    f'<div class="pct" style="color:{BLUE}">{r["P_champion"]:.0%}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("### 🏆 Title odds")
    lc, rc = st.columns([1, 1])
    lc.markdown(prob_rows(adv), unsafe_allow_html=True)
    rc.dataframe(advancement_table(adv), use_container_width=True, height=430, hide_index=True)

    st.markdown("### 🥊 Series matchups")
    rnd = st.selectbox("Round", [(1, "First round"), (2, "Conf semifinals"),
                                 (3, "Conf finals"), (4, "NBA Finals")],
                       format_func=lambda t: t[1])[0]
    st.caption("First round is certain; later rounds show the most likely opponents.")
    st.markdown(matchup_cards(mu, rnd), unsafe_allow_html=True)


# ---------------- sidebar ----------------
st.sidebar.markdown("## 🏀 Mode")
mode = st.sidebar.radio("", ["Historical season", "Upload a season (JSON)", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Walk-forward · no leakage · 1996–2025")

if mode == "Historical season":
    seasons = available_seasons()
    labels = {label_with_title(y): y for y in seasons}
    pick = st.sidebar.selectbox("Season", list(labels.keys())[::-1])
    year = labels[pick]
    n_sims = st.sidebar.slider("Monte Carlo sims", 1000, 10000, 5000, 1000)
    with st.spinner(f"Training on seasons before {season_label(year)} and simulating…"):
        res = predict_season(year, n_sims=n_sims)
        real = real_outcome(year)
    render_prediction(res["advancement"], res["matchups"], real)
    with st.expander("📊 Team strengths this season"):
        st.dataframe(team_strengths(year), use_container_width=True, hide_index=True)

elif mode == "Upload a season (JSON)":
    st.markdown("### 📤 Upload a finished regular season")
    st.caption("Drop a JSON of per-player & per-team values; the full pipeline runs "
               "and predicts the playoffs. Format: `data/sample_new_season.json`.")
    up = st.file_uploader("Season JSON", type="json")
    use_sample = st.checkbox("Use the bundled 2023-24 example", value=not up)
    n_sims = st.slider("Monte Carlo sims", 1000, 10000, 5000, 1000)
    payload = None
    if up is not None:
        payload = json.load(up)
    elif use_sample and (ROOT / "data" / "sample_new_season.json").exists():
        payload = json.loads((ROOT / "data" / "sample_new_season.json").read_text())
    if payload is not None:
        from src.ingest.new_season import predict_new_season
        with st.spinner("Running the pipeline and simulating…"):
            res = predict_new_season(payload, n_sims=n_sims)
        render_prediction(res["advancement"], res["matchups"])
    else:
        st.info("Upload a JSON or tick the example box.")

else:
    st.markdown("""
### How a team wins

This project tests a thesis: **what actually makes a team win in the NBA?**
Who wins a *series* is dominated by net rating and home court — but the thesis
factors (three-point shooting, size, playoff experience, role quality) act
**upstream**: they **build** the high-net-rating team that then wins. Player
features explain ~62% of net rating.
""")
    for fn in ("causal_chain.png", "rating_drivers.png", "champion_probabilities.png"):
        p = ROOT / "reports" / "figures" / fn
        if p.exists():
            st.image(str(p), use_container_width=True)
