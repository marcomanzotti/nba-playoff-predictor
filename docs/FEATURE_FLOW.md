# How the features combine — from player to matchup

This document explains **exactly** how information flows from individual players
all the way to a series win probability. It answers the question: *"are player
stats weighted by minutes, and how do they mix with team stats and the matchup?"*

```
PLAYER-SEASON  ──weighting──►  TEAM-SEASON  ──difference A-B──►  SERIES (matchup)  ──►  P(A beats B)
   (technical,                   (T1..T5 +                        (d_*, H2H,            XGBoost
    career,                       team stats +                     HOME_COURT)          model
    physical, level)              home/away splits)
```

---

## Level 1 — The PLAYER (one row per player, per season)

For every player in every season we collect 4 blocks:

| Block | Example fields | File |
|-------|----------------|------|
| **Technical** | points, assists, TS%, USG%, PIE, three-point (volume/efficiency) | `player_technical_rs/po` |
| **Career** | years of experience, homegrown 0-4, playoff experience weighted by depth, titles | `player_career` |
| **Physical/athletic** | height, wingspan, vertical, agility (real or proxy if Combine is missing) | `player_physical` |
| **Level/impact** | superstar / all-star / starter / role / bench / bench-warmer | `player_level` |

---

## Level 2 — From PLAYER to TEAM: **minutes weighting**

This is the heart of the question. A team is not the sum of its players: it is a
**minutes-weighted average**. A role player who plays 30 minutes counts more than
a star who plays 12.

For every team-season, for each player metric:

```
team_value = Σ ( player_value × player_weight )

where   player_weight = player_MIN_TOT / team_total_MIN_TOT
```

Concrete examples (in `team_features.py`):

- **T2 size**: `team_height = mean(player_height, minutes-weighted)` → a team that
  gives heavy minutes to tall players reads as "bigger".
- **T1 homegrown**: `Σ (min_share × homegrown_score/4)` → how much of the playing
  time is produced by players developed in-house.
- **T3 playoff experience**: each player's experience is itself **weighted by
  actual playoff minutes**: `round_depth × (playoff_minutes/100)`. So a star at
  40 min × 13 games in a deep run accumulates far more credit than a bench player
  at 2 min, even if the team reached the same round. The team value is then the
  minutes-weighted average.
- **T5 involvement**: minutes/points share of non-stars + entropy of the
  contribution distribution (how "spread out" vs concentrated on the stars).

### Weighting on PLAYOFF minutes

In the playoffs rotations shrink (~8 players). We therefore keep playoff stats
and minutes (`player_technical_po`), used leakage-free: a player's playoff
experience/output enters as the **history of prior seasons** (never the playoffs
of the season being predicted).

---

## Level 3 — The full TEAM

To the player-aggregated team we add **official team stats** (which don't come
from individuals):

- record (W, L, W%), offensive/defensive rating, net rating, pace;
- team three-point shooting (volume, %);
- **home/away splits**: home win%, road win%, `HOME_SPLIT` (how much the team
  depends on home court), home/road net rating.

Result: **one row per team-season with ~76 columns** (`team_season_features`).

---

## Features PER ROLE BAND (backcourt / wing / frontcourt)

A single average hides nuance: "physical bigs + small shooters" and "uniform
average height" can share the **same mean** yet be opposite teams, with different
weaknesses by position. So we compute size, shooting, experience and level
**separately for 3 bands** (`BAND_back_*`, `BAND_wing_*`, `BAND_fron_*`). This is
how **matchups by role** emerge: you struggle against a team that is physical **in
the role where you are weak**, not in the abstract.

## No redundant features (for XGBoost)

XGBoost captures interactions and non-linear relationships on its own:
near-duplicate features don't help and they **dilute feature importance** (fatal
for reading the thesis). A selection step (`select_features.py`) keeps only an
**uncorrelated** set (|corr| ≤ 0.85): out go duplicates like `W`/`L`/`W_PCT`
(we keep `NET_RATING`) and the era-zscore versions (redundant because the A−B
difference already normalizes context). From ~75 down to ~52 features, one per
concept.

---

## Level 4 — The MATCHUP: from two teams to a series

A series pits team A **against** team B. The model doesn't look at absolute
values but at the **relative difference** between the two (what matters in a head
-to-head):

```
for each team feature X:   d_X = X(A) − X(B)
```

To these differences we add two **matchup-specific** features:

| Feature | Meaning |
|---------|---------|
| `H2H_DIFF` | regular-season head-to-head difference (did A beat B more often?) — captures the "matchup": a team can beat a stronger one on paper |
| `HOME_COURT` | +1 if A has home court (better seed → plays games 1,2,5,7 at home), −1 if B does. **Very strong**: historically the home-court team wins ~74% of series |

The final vector `[d_W..., d_T1..d_T5, ..., H2H_DIFF, HOME_COURT]` is what the
model receives to output **P(A wins the series)**.

### Guaranteed symmetry

By construction all these features are **antisymmetric**: swapping A and B flips
the sign. We also train on the mirrored rows and average the two views at
prediction time, so `P(A>B) = 1 − P(B>A)` always.

---

## Level 5 — From series to BRACKET

The series model is the building block. A Monte Carlo simulator applies it to
every matchup of the conference bracket thousands of times → from this we get:

- **per team**: P(advance round 1), P(round 2), P(conf finals), P(title);
- **per matchup**: the series win % of **both** teams.

---

## The causal chain (key finding)

The series model showed that **who wins a series** is dominated by record / net
rating / home court — which is almost circular (the better team wins). The deeper
question is *what makes a team strong?* A separate model (`what_builds_rating.py`)
predicts **net rating from player features alone** and finds that the thesis
factors act **upstream**:

```
   T4 three-point  ┐
   T2 size         │
   T3 experience   ├──►  HIGH NET RATING  ──►  wins the playoffs
   role quality    │      (+ home court)
   star power      ┘
```

Player features explain ~62% of net rating. The thesis was right, but it acts
**upstream**: homegrown/size/shooting/experience don't win the series directly —
they **build the dominant team** (high net rating), and that team wins.

---

## In one sentence

> **Player** stats are **weighted by minutes** (regular season, and playoff
> minutes for experience) to form the **team** profile; the team profile combines
> with record, ratings and home/away splits; two teams are compared by
> **difference** plus `H2H` and `HOME_COURT` (the matchup); the model turns this
> into **P(series win)**, which Monte Carlo propagates into round-by-round and
> title probabilities.
