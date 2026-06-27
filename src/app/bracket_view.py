"""Rendering del BRACKET stile tabellone TV (Sporting News).

Disegna il tabellone come HTML: due conference ai lati, le colonne dei round che
convergono verso il trofeo al centro, loghi grandi, vincitore evidenziato e
connettori tra i round. NON contiene logica di modello: consuma solo i
`SeriesView` prodotti da InteractiveBracket.series(); i controlli what-if
(radio) restano in app.py.

Palette coerente con app.py.
"""
from __future__ import annotations

from src.app.teams_meta import TROPHY_URL
from src.app.teams_meta import color as team_color
from src.app.teams_meta import logo_url

GOLD, GREEN, INK, MUTED = "#FFC857", "#2ECC71", "#E6EDF3", "#8B949E"
PANEL, LINE, FORCED = "#161B22", "#2a3550", "#FFC857"


def bracket_css() -> str:
    return f"""
<style>
  .bk {{ display:flex; align-items:stretch; justify-content:center; gap:0;
         width:100%; overflow-x:auto; padding:.5rem 0 1rem; }}
  .bk-col {{ display:flex; flex-direction:column; justify-content:space-around;
             gap:.4rem; min-width:150px; }}
  .bk-col.r2 {{ min-width:150px; }}
  .bk-col.fin {{ justify-content:center; min-width:190px; }}
  .bk-rnd {{ text-align:center; color:{MUTED}; font-size:.62rem; font-weight:800;
             text-transform:uppercase; letter-spacing:1.5px; margin-bottom:.3rem; }}
  .ser {{ background:{PANEL}; border:1px solid {LINE}; border-radius:10px;
          padding:.3rem .4rem; position:relative; }}
  .ser.forced {{ border-color:{FORCED}; box-shadow:0 0 0 1px {FORCED} inset; }}
  .tl {{ display:flex; align-items:center; gap:.4rem; padding:.16rem .1rem; border-radius:6px; }}
  .tl img {{ width:24px; height:24px; object-fit:contain; flex:0 0 24px; }}
  .tl .nm {{ font-weight:800; font-size:.82rem; color:{INK}; flex:1;
             white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .tl .pp {{ font-weight:800; font-size:.78rem; width:38px; text-align:right; }}
  .tl.win {{ background:linear-gradient(90deg, rgba(46,204,113,.16), transparent); }}
  .tl.win .pp {{ color:{GREEN}; }}
  .tl.lose {{ opacity:.4; }}
  .tl.tbd .nm {{ color:{MUTED}; font-style:italic; }}
  .whatif {{ position:absolute; top:-7px; right:6px; background:{FORCED}; color:#111;
             font-size:.52rem; font-weight:900; padding:.04rem .35rem; border-radius:8px;
             letter-spacing:.5px; }}
  .fin-card {{ background:linear-gradient(160deg,#20283a,{PANEL}); border:1px solid {LINE};
               border-radius:14px; padding:.7rem .5rem; text-align:center; }}
  .fin-card img.trophy {{ height:52px; filter:drop-shadow(0 0 12px {GOLD}); }}
  .fin-card .champ-logo {{ height:60px; object-fit:contain; margin:.3rem 0; }}
  .fin-card .champ-nm {{ font-size:1.3rem; font-weight:900; color:{GOLD}; }}
  .fin-card .lbl {{ color:{MUTED}; font-size:.6rem; text-transform:uppercase;
                    letter-spacing:1.5px; }}
</style>"""


def _team_line(team, p, is_win, tbd=False):
    if tbd or team is None:
        return ('<div class="tl tbd"><span class="nm">— da decidere —</span>'
                '<span class="pp"></span></div>')
    cls = "win" if is_win else "lose"
    return (f'<div class="tl {cls}"><img src="{logo_url(team)}">'
            f'<span class="nm">{team}</span><span class="pp">{p:.0%}</span></div>')


def _series_html(s) -> str:
    forced = " forced" if s.forced_winner else ""
    tag = '<span class="whatif">WHAT-IF</span>' if s.forced_winner else ""
    win = s.modal_winner
    tbd = s.team_a is None or s.team_b is None
    return (f'<div class="ser{forced}">{tag}'
            f'{_team_line(s.team_a, s.p_a, win == s.team_a, tbd)}'
            f'{_team_line(s.team_b, s.p_b, win == s.team_b, tbd)}</div>')


def _col(title, series_list) -> str:
    cells = "".join(_series_html(s) for s in series_list)
    return f'<div class="bk-col"><div class="bk-rnd">{title}</div>{cells}</div>'


def render_bracket_html(series: list, champion: str | None) -> str:
    """Tabellone completo come stringa HTML.

    Ordine colonne (da sinistra a destra, convergenti al centro):
      West R1 | West R2 | West CF || FINALS+trofeo || East CF | East R2 | East R1
    """
    east = [s for s in series if s.conference == "East"]
    west = [s for s in series if s.conference == "West"]
    finals = next((s for s in series if s.sid == "FINALS"), None)

    def rnd(lst, r):
        return [s for s in lst if s.round == r]

    # colonna Finals con trofeo + campione
    champ_html = ""
    if champion:
        champ_html = (f'<img class="champ-logo" src="{logo_url(champion)}">'
                      f'<div class="champ-nm">{champion}</div>')
    else:
        champ_html = '<div class="champ-nm" style="color:#8B949E">?</div>'
    fin_series = _series_html(finals) if finals else ""
    fin_col = (f'<div class="bk-col fin"><div class="bk-rnd">NBA Finals</div>'
               f'<div class="fin-card"><img class="trophy" src="{TROPHY_URL}">'
               f'<div class="lbl">Champion</div>{champ_html}</div>{fin_series}</div>')

    html = ['<div class="bk">']
    # West (sinistra): R1 -> R2 -> CF
    html.append(_col("West · R1", rnd(west, 1)))
    html.append(_col("West · Semis", rnd(west, 2)))
    html.append(_col("West · Finals", rnd(west, 3)))
    # centro
    html.append(fin_col)
    # East (destra): CF -> R2 -> R1 (specchiato)
    html.append(_col("East · Finals", rnd(east, 3)))
    html.append(_col("East · Semis", rnd(east, 2)))
    html.append(_col("East · R1", rnd(east, 1)))
    html.append("</div>")
    return bracket_css() + "".join(html)
