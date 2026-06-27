"""NBA Playoff Predictor — interactive dashboard (real NBA imagery).

Run with:  streamlit run app.py

Highlights:
  - Real NBA team logos (official NBA CDN, ESPN fallback) and the real Larry
    O'Brien trophy.
  - A fully interactive what-if bracket: override any series winner and watch the
    whole bracket and the title odds recompute, conditioned on your picks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.app.engine import (available_seasons, interactive_for, real_outcome,
                            team_strengths)
from src.app.teams_meta import NBA_LOGO_URL, TROPHY_URL
from src.app.teams_meta import color as team_color
from src.app.teams_meta import logo_url
from src.season_labels import label_with_title, season_label

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="NBA Playoff Predictor", page_icon="🏀", layout="wide")

ORANGE, GOLD, BLUE, GREEN = "#FF6B35", "#FFC857", "#1D9BF0", "#2ECC71"
INK, MUTED, PANEL, BG = "#E6EDF3", "#8B949E", "#161B22", "#0B0E14"
FORCED = "#FFC857"

st.markdown(f"""
<style>
  .stApp {{ background: radial-gradient(1200px 600px at 50% -10%, #1a2030 0%, {BG} 55%); }}
  #MainMenu, footer {{ visibility: hidden; }}
  h1,h2,h3,h4 {{ color:{INK}; font-weight:800; letter-spacing:.3px; }}
  .hero {{ text-align:center; padding: 1.2rem 1rem .4rem; }}
  .hero img.trophy {{ height:74px; filter: drop-shadow(0 0 16px {GOLD}); }}
  .hero h1 {{ font-size:2.6rem; margin:.2rem 0 0; background:linear-gradient(90deg,{GOLD},{ORANGE});
              -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
  .hero p {{ color:{MUTED}; margin:.2rem 0 0; font-size:1rem; }}
  .champ-card {{ background:linear-gradient(160deg,#20283a,{PANEL}); border:1px solid #2a3550;
                 border-radius:18px; padding:1.2rem; text-align:center;
                 box-shadow:0 8px 30px rgba(0,0,0,.4); }}
  .champ-card .lbl {{ color:{MUTED}; text-transform:uppercase; font-size:.75rem; letter-spacing:2px; }}
  .champ-card .team {{ font-size:2rem; font-weight:900; color:{INK}; }}
  .champ-card .pct {{ font-size:2.4rem; font-weight:900; color:{GOLD}; }}
  .champ-card img {{ height:78px; object-fit:contain; }}
  .prob-row {{ display:flex; align-items:center; gap:.6rem; padding:.35rem .5rem;
               border-radius:10px; margin-bottom:.25rem; background:{PANEL}; }}
  .prob-row img {{ width:30px; height:30px; object-fit:contain; }}
  .prob-row .abbr {{ width:46px; font-weight:800; color:{INK}; }}
  .bar-wrap {{ flex:1; background:#0d1117; border-radius:6px; height:20px; overflow:hidden; }}
  .bar {{ height:100%; border-radius:6px; }}
  .prob-row .val {{ width:52px; text-align:right; font-weight:800; color:{INK}; }}
  /* bracket */
  .series {{ background:{PANEL}; border:1px solid #232c40; border-radius:12px;
             padding:.45rem .6rem; margin:.32rem 0; }}
  .series.forced {{ border:1px solid {FORCED}; box-shadow:0 0 0 1px {FORCED} inset; }}
  .series .rnd {{ color:{MUTED}; font-size:.66rem; text-transform:uppercase; letter-spacing:1px; }}
  .team-line {{ display:flex; align-items:center; gap:.5rem; padding:.12rem 0; }}
  .team-line img {{ width:26px; height:26px; object-fit:contain; }}
  .team-line .nm {{ font-weight:800; flex:1; }}
  .team-line .p {{ font-weight:800; width:46px; text-align:right; }}
  .team-line.win .nm {{ color:{INK}; }}
  .team-line.lose {{ opacity:.45; }}
  .team-line.win .p {{ color:{GREEN}; }}
  .tag {{ display:inline-block; font-size:.6rem; font-weight:800; padding:.05rem .4rem;
          border-radius:10px; background:{FORCED}; color:#111; margin-left:.3rem; }}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero">
  <img class="trophy" src="{TROPHY_URL}">
  <h1>NBA PLAYOFF PREDICTOR</h1>
  <p>Regular-season data → series-by-series playoff probabilities · walk-forward validated · 1996–2025</p>
</div>
""", unsafe_allow_html=True)


def _img(abbr, size=40):
    """Real NBA team logo (ESPN-CDN PNG; renders reliably in-browser)."""
    return f'<img src="{logo_url(abbr)}" width="{size}">'


# ---------------- title odds (per-team bars + table) ----------------
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


def advancement_table(adv: pd.DataFrame) -> pd.DataFrame:
    d = adv.copy()
    for c in ["P_pass_R1", "P_pass_R2", "P_pass_ConfFinals", "P_champion"]:
        d[c] = (d[c] * 100).round(1)
    return d.rename(columns={"TEAM": "Team", "P_pass_R1": "Round 2 %",
                             "P_pass_R2": "Conf SF %", "P_pass_ConfFinals": "Conf Finals %",
                             "P_champion": "Champion %"})[
        ["Team", "Round 2 %", "Conf SF %", "Conf Finals %", "Champion %"]]


# ---------------- interactive bracket ----------------
def _series_card(s, key_prefix: str):
    """Render one series card + an override radio. Returns the chosen winner
    ('Model' means: no override)."""
    a, b = s.team_a, s.team_b
    forced_cls = " forced" if s.forced_winner else ""
    win = s.modal_winner
    def line(team, p, is_win):
        cls = "win" if is_win else "lose"
        return (f'<div class="team-line {cls}">{_img(team,26)}'
                f'<span class="nm">{team}</span><span class="p">{p:.0%}</span></div>')
    tag = '<span class="tag">WHAT-IF</span>' if s.forced_winner else ""
    body = (f'<div class="series{forced_cls}">'
            f'<div class="rnd">{s.round_name}{tag}</div>'
            f'{line(a, s.p_a, win==a)}{line(b, s.p_b, win==b)}</div>')
    st.markdown(body, unsafe_allow_html=True)
    # override control
    opts = ["Model", a, b]
    cur = s.forced_winner if s.forced_winner in (a, b) else "Model"
    pick = st.radio("force winner", opts, index=opts.index(cur),
                    key=f"{key_prefix}_{s.sid}", horizontal=True,
                    label_visibility="collapsed")
    return None if pick == "Model" else pick


def render_bracket(ib, year: int):
    """Interactive what-if bracket. Overrides live in session_state['ov']."""
    ov = st.session_state.setdefault("ov", {})

    top = st.columns([3, 1])
    top[0].markdown("### 🗺️ Interactive bracket — override any winner to see the what-if")
    if top[1].button("↩︎ Reset what-ifs", use_container_width=True):
        # clear stored overrides + the radio widget states
        for k in list(st.session_state.keys()):
            if k == "ov" or k.startswith(f"ovr{year}_"):
                del st.session_state[k]
        ov = st.session_state.setdefault("ov", {})
        st.rerun()

    # First pass: read current series with existing overrides so the radios
    # show the right teams; the radios themselves then update `ov`.
    series = ib.series(overrides=ov)
    by = {s.sid: s for s in series}

    east = [s for s in series if s.conference == "East"]
    west = [s for s in series if s.conference == "West"]
    finals = next(s for s in series if s.sid == "FINALS")

    def col_for_round(series_list, rnd):
        return [s for s in series_list if s.round == rnd]

    new_ov = {}
    cE, cMid, cW = st.columns([1.1, 1, 1.1])
    with cE:
        st.markdown("#### 🟢 East")
        for rnd, name in [(1, "First round"), (2, "Conf semis"), (3, "Conf finals")]:
            for s in col_for_round(east, rnd):
                pick = _series_card(by[s.sid], f"ovr{year}")
                if pick:
                    new_ov[s.sid] = pick
    with cW:
        st.markdown("#### 🔵 West")
        for rnd, name in [(1, "First round"), (2, "Conf semis"), (3, "Conf finals")]:
            for s in col_for_round(west, rnd):
                pick = _series_card(by[s.sid], f"ovr{year}")
                if pick:
                    new_ov[s.sid] = pick
    with cMid:
        st.markdown("#### 🏆 Finals")
        st.markdown(f'<img src="{TROPHY_URL}" style="height:64px;display:block;'
                    f'margin:.2rem auto;filter:drop-shadow(0 0 14px {GOLD})">',
                    unsafe_allow_html=True)
        pick = _series_card(finals, f"ovr{year}")
        if pick:
            new_ov["FINALS"] = pick
        # show the resolved champion of the (possibly forced) bracket
        champ = finals.modal_winner
        if champ:
            st.markdown(
                f'<div class="champ-card"><div class="lbl">Bracket winner</div>'
                f'{_img(champ,72)}<div class="team">{champ}</div></div>',
                unsafe_allow_html=True)

    # If overrides changed, store and rerun so downstream rounds re-resolve.
    if new_ov != ov:
        st.session_state["ov"] = new_ov
        st.rerun()
    return new_ov


def render_prediction(adv, mu, real=None):
    champ = adv.iloc[0]
    cols = st.columns([1, 1, 1])
    with cols[0]:
        st.markdown(
            f'<div class="champ-card"><div class="lbl">Predicted champion</div>'
            f'{_img(champ["TEAM"],86)}<div class="team">{champ["TEAM"]}</div>'
            f'<div class="pct">{champ["P_champion"]:.0%}</div></div>', unsafe_allow_html=True)
    if real and real.get("champion"):
        with cols[1]:
            ok = champ["TEAM"] == real["champion"]
            st.markdown(
                f'<div class="champ-card"><div class="lbl">Actual champion</div>'
                f'{_img(real["champion"],86)}'
                f'<div class="team">{real["champion"]}</div>'
                f'<div class="pct" style="color:{GREEN if ok else ORANGE}">'
                f'{"✅ HIT" if ok else "❌ MISS"}</div></div>', unsafe_allow_html=True)
        with cols[2]:
            runner = adv.iloc[1]
            st.markdown(
                f'<div class="champ-card"><div class="lbl">Top contender #2</div>'
                f'{_img(runner["TEAM"],86)}<div class="team">{runner["TEAM"]}</div>'
                f'<div class="pct" style="color:{BLUE}">{runner["P_champion"]:.0%}</div></div>',
                unsafe_allow_html=True)
    else:
        for i, c in enumerate(cols[1:], start=1):
            r = adv.iloc[i]
            with c:
                st.markdown(
                    f'<div class="champ-card"><div class="lbl">Contender #{i+1}</div>'
                    f'{_img(r["TEAM"],86)}<div class="team">{r["TEAM"]}</div>'
                    f'<div class="pct" style="color:{BLUE}">{r["P_champion"]:.0%}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("### 🏆 Title odds")
    lc, rc = st.columns([1, 1])
    lc.markdown(prob_rows(adv), unsafe_allow_html=True)
    rc.dataframe(advancement_table(adv), use_container_width=True, height=430, hide_index=True)


# ---------------- sidebar ----------------
st.sidebar.markdown(f'<img src="{NBA_LOGO_URL}" style="height:54px;margin:.2rem 0 1rem">',
                    unsafe_allow_html=True)
st.sidebar.markdown("## 🏀 Mode")
mode = st.sidebar.radio("mode", ["Historical season", "Upload a season (JSON)", "About"],
                        label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Walk-forward · no leakage · 1996–2025")

if mode == "Historical season":
    seasons = available_seasons()
    labels = {label_with_title(y): y for y in seasons}
    pick = st.sidebar.selectbox("Season", list(labels.keys())[::-1])
    year = labels[pick]
    n_sims = st.sidebar.slider("Monte Carlo sims", 1000, 10000, 5000, 1000)

    # reset what-ifs whenever the season changes
    if st.session_state.get("season_year") != year:
        st.session_state["season_year"] = year
        for k in list(st.session_state.keys()):
            if k == "ov" or k.startswith("ovr"):
                del st.session_state[k]

    with st.spinner(f"Training on seasons before {season_label(year)} and simulating…"):
        ib = interactive_for(year)
        real = real_outcome(year)

    # interactive bracket (reads/writes session_state['ov'])
    overrides = render_bracket(ib, year)

    # title odds conditioned on the current what-if overrides
    st.markdown("---")
    if overrides:
        st.info("🔮 Showing odds **conditioned on your what-if picks**. "
                "Forced series are locked; everything downstream is re-simulated.")
    adv = ib.title_odds(overrides=overrides, n_sims=n_sims)
    render_prediction(adv, None, None if overrides else real)

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
    st.markdown(f'<img src="{NBA_LOGO_URL}" style="height:48px;margin-bottom:1rem">',
                unsafe_allow_html=True)
    st.markdown("""
### How a team wins

This project tests a thesis: **what actually makes a team win in the NBA?**
Who wins a *series* is dominated by net rating and home court — but the thesis
factors (three-point shooting, size, playoff experience, role quality) act
**upstream**: they **build** the high-net-rating team that then wins. Player
features explain ~62% of net rating.

**Interactive what-if bracket:** in *Historical season* mode you can override
any series winner — e.g. *"what if Atlanta had beaten New York?"* — and the
whole bracket plus every team's title odds re-compute, conditioned on your pick.
""")
    for fn in ("causal_chain.png", "rating_drivers.png", "champion_probabilities.png"):
        p = ROOT / "reports" / "figures" / fn
        if p.exists():
            st.image(str(p), use_container_width=True)
