# NBA Playoff Predictor — Specifica di Progetto

> **Stato:** BOZZA da approvare
> **Data:** 2026-06-26
> **Obiettivo:** Costruire un modello statistico modulare che, dati i risultati e le statistiche di regular season, predice i playoff NBA (singola serie → bracket completo → probabilità di stadio → campione), e che **testa la tesi** che in NBA si vinca grazie a: giocatori cresciuti in casa, fisico/atletismo, esperienza playoff, tiro da tre e alto coinvolgimento dei non-superstar.

---

## 0. La Tesi (il cuore del progetto)

La domanda di ricerca non è solo "chi vince?", ma **"cosa fa vincere?"**. La tesi è che la vittoria NBA sia spiegata da 5 fattori strutturali:

| # | Fattore | Misura operativa (bozza) |
|---|---------|--------------------------|
| T1 | **Giocatori "hometown"** (cresciuti in casa) | Quota di minuti/valore prodotto da giocatori draftati e sviluppati dalla franchigia (scala graduata, vedi §3.3) |
| T2 | **Fisico e atletismo** | Indici aggregati di taglia (altezza, wingspan, reach, peso) e atletismo (vertical, agility, sprint) pesati per minuti |
| T3 | **Esperienza playoff** | Minuti/partite playoff accumulati in carriera dai giocatori del roster, pesati per profondità raggiunta |
| T4 | **Tiro da tre** | Volume + efficienza da 3 a livello squadra e distribuzione tra i giocatori |
| T5 | **Coinvolgimento dei non-superstar** | Quanto il valore prodotto è distribuito oltre le 1-2 stelle (entropia/Gini del contributo nel roster) |

**Come testiamo la tesi (non solo predire):**
- **A) Predizione**: le 5 feature-tesi entrano nel modello insieme alle altre.
- **B) Analisi causale/esplicativa dedicata**: feature importance, SHAP, e **modelli ablati** (con/senza le feature-tesi) per isolare il loro effetto reale.
- **C) Modello "solo-tesi" vs benchmark**: un modello costruito SOLO con T1–T5, confrontato con il modello completo, per quantificare quanto la tesi spiega da sola.

> Senza la parte B/C avremmo solo un predittore; con essa abbiamo una **dimostrazione** (per quanto possibile da dati osservazionali — vedi §8 sui limiti causali).

---

## 1. Output del progetto

1. **Modello modulare**
   - *Mattone base:* **Modello di singola serie** — P(squadra A batte squadra B in una serie al meglio delle 7), dati i due roster e i contesti.
   - *Derivati:* simulazione Monte Carlo del **bracket completo**, **probabilità di ogni stadio** per ogni squadra (playoff → 2° turno → finale conf → finale NBA → titolo), e **campione**.
2. **Backtest storico**: per ogni stagione di test, bracket predetto vs reale, con metriche (serie azzeccate, round azzeccati, log-loss, Brier score).
3. **Suite di visualizzazioni** per dimostrare la tesi (vedi §7).
4. **Dataset riproducibili** (un file per giocatore-stagione, uno per squadra-stagione, head-to-head).

---

## 2. Dati

### 2.1 Fonti
- **Primaria:** `nba_api` (wrapper Python di stats.nba.com) — box score, advanced stats, head-to-head, draft, combine, lineup. Gratis ma **rate-limited** e occasionalmente instabile (gestiremo retry/caching).
- **Secondaria:** **Basketball-Reference** (scraping mirato) solo per riempire i buchi storici e dati Combine non disponibili via API.
- **Caching locale obbligatorio**: ogni chiamata salvata su disco (parquet/sqlite) per non ri-scaricare e non superare i rate limit.

### 2.2 Orizzonte temporale
- **Target ideale:** ultimi **30 anni** (≈ stagione 1996-97 in poi, quando stats.nba.com ha advanced stats e play-by-play affidabili).
- **Fallback:** 15 anni se la copertura/qualità a 30 risulta insufficiente.
- **Decisione data-driven:** una **Fase 0 di audit copertura** stabilisce fin dove possiamo andare (vedi §6 Fase 0).

### 2.3 Granularità richiesta
- **Giocatore × stagione** (una riga per ogni giocatore per ogni anno): stat regular season, stat playoff (di quell'anno e accumulate in carriera), misure fisiche, atletismo, dati di carriera.
- **Squadra × stagione**: tutte le team stats, record complessivo, **record testa-a-testa** vs ogni altra squadra in regular season.
- **Serie playoff** (storico): chi vs chi, round, esito, gara-per-gara (per il modello di serie).

---

## 3. Feature Engineering — Livello GIOCATORE

Una riga per **giocatore × stagione**, con quattro blocchi.

### 3.1 Blocco FISICO / ATLETICO
- Misure: altezza (con/senza scarpe), **wingspan**, standing reach, peso, body fat, hand size.
- Atletismo (Combine): max vertical, no-step vertical, lane agility, 3/4 sprint, bench.
- **Problema noto:** molti giocatori non hanno dati Combine (undrafted, internazionali, pre-Combine).
  → **Soluzione (clustering + proxy)**, vedi §3.5.

### 3.2 Blocco TECNICO (statistiche di gioco)
- Box score classico (per game e per 36/per 100 possessi): punti, rimbalzi, assist, palle perse, tiri, %.
- **Advanced:** TS%, eFG%, USG%, PER, BPM, OBPM/DBPM, VORP, WS, ON/OFF, RAPM-like se disponibile.
- **Shooting profile** dettagliato (per la tesi T4): volume e % da 3, frequenza per zona, catch&shoot vs pull-up se disponibili.
- Versione **regular season** e versione **playoff** (vedi §3.4 per l'uso anti-leakage).

### 3.3 Blocco CARRIERA
- Anni di esperienza, età (con il caveat **"LeBron a 41 ≠ LeBron a 24"** → vedi §3.6 aging curve).
- **Hometown score (T1) — scala graduata** (non sì/no):
  - `4` = draftato dalla franchigia **e sempre rimasto**
  - `3` = draftato dalla franchigia, andato via e **tornato**
  - `2` = acquisito ma **di lunga data** (≥ N stagioni nella squadra)
  - `1` = acquisito di **recente**
  - `0` = primo anno nella squadra / appena arrivato
  - (Soglie esatte di N definite in implementazione e documentate.)
- **Esperienza playoff (T3):** partite/minuti playoff in carriera, **pesati per profondità** raggiunta (1° turno < semifinale conf < finale conf < finale NBA < titolo).
- "Deep playoff experience" per stadio: conteggio separato per ogni stadio raggiunto.

### 3.4 Statistiche PLAYOFF — uso anti-leakage
> Confermato: si intendono le **stat dei PLAYOFF** (non preseason/offseason). Nei playoff le rotazioni si accorciano (~8 giocatori) e il gioco cambia.

Usate in **due modi**, entrambi senza leakage:
1. **Storico di rendimento playoff** del giocatore (anni **precedenti** a quello predetto): "quanto rende questo giocatore nei playoff rispetto alla regular season" come feature.
2. **Modellazione dell'accorciamento delle rotazioni**: nei playoff il valore si concentra sui top-8; ri-pesiamo il contributo del roster di conseguenza nelle proiezioni.

> **Regola d'oro:** non si usano MAI le statistiche dei playoff **dell'anno che stiamo predicendo** come input. Quelle sono il target/validazione.

### 3.5 Clustering #1 — Profilo di gioco + proxy atletismo
- Addestriamo un clustering (es. K-Means/GMM/HDBSCAN su feature standardizzate) **sui giocatori CON dati Combine**.
- Per i giocatori **senza** Combine: li proiettiamo nello spazio (stile di gioco + fisico noto) e assegniamo **valori proxy** atletici dai vicini/cluster.
- Serve un **campione ampio di giocatori Combine** per cluster robusti → motivo in più per puntare a 30 anni.
- Output: `player_archetype` (es. "lob-threat big", "3&D wing", "shifty guard"...) + atletismo imputato con flag `is_imputed`.

> **Copertura Combine (da audit Fase 0):** i dati Combine via `nba_api` sono pieni e stabili **dal 2000** (~46-83 giocatori/anno; anthropometrics quasi sempre presenti, atletismo ~80%). Per le 4 stagioni **1996-99** la Combine non esiste → quei giocatori vengono trattati come "senza Combine" e ricevono valori proxy **con lo stesso meccanismo di clustering** usato per gli undrafted. Flag `is_imputed=True` su tutti i proxy (sia pre-2000 sia undrafted), così il modello può pesarne l'incertezza.

### 3.6 Clustering #2 — Livello / Impatto (per la tesi T5)
Categorie graduate di "livello" del giocatore in quella stagione:
- **Superstar → All-Star → Starter di qualità → Role player → Bench → Bench-warmer** (e sottocategorie se i dati lo permettono).
- Basato su advanced stats + minuti + usage + impatto.
- Serve a: (a) feature di "qualità roster", (b) **misurare il coinvolgimento dei non-superstar (T5)** — es. quanta produzione viene da giocatori sotto il livello All-Star.

### 3.7 Aging curve
- L'effetto dell'età non è lineare: modelliamo curve di invecchiamento per archetipo, così l'**età** corregge le proiezioni (peak ~27-29, declino dopo).

---

## 4. Feature Engineering — Livello SQUADRA

Una riga per **squadra × stagione**:
- Tutte le **team stats** (offensive/defensive rating, pace, net rating, four factors, rating per quarto, clutch stats).
- **Record** complessivo (W-L, home/away, vs playoff teams, in clutch).
- **Record testa-a-testa** vs ogni altra squadra in regular season → **matrice matchup** (cattura l'idea: "SF batte OKC ma perde con NY").
- **Aggregati dal livello giocatore** (T1–T5 a livello squadra): hometown share, indici fisico/atletismo pesati per minuti, esperienza playoff totale del roster, profilo di tiro da 3, distribuzione del contributo (entropia/Gini).
- **Roster construction:** distribuzione dei "livelli" (quante superstar/starter/role player), continuità del roster anno su anno.

---

## 5. Modello

### 5.1 Modello base: singola serie (best-of-7)
- Input: rappresentazione squadra A, squadra B, **feature di matchup** (incluso head-to-head e differenze relative), contesto (vantaggio campo/seed).
- Output: P(A vince la serie).
- Candidati: gradient boosting (XGBoost/LightGBM) come baseline forte; possibile estensione a modello di "partita" → simulazione serie.
- **Simmetria:** il modello deve dare P(A>B) = 1 − P(B>A) (feature antisimmetriche / data augmentation con squadre scambiate).

### 5.2 Da serie a bracket
- Dato il seeding reale, simulazione **Monte Carlo** del bracket (N migliaia di simulazioni) → distribuzione degli esiti per ogni squadra → probabilità di stadio e di titolo.

### 5.3 Validazione — Split temporale 20 / 5 / 5 + walk-forward espandente
Su un orizzonte di **30 stagioni**, divisione temporale in tre blocchi (in ordine cronologico, mai mescolati):

| Blocco | Stagioni | Ruolo |
|--------|----------|-------|
| **Training** | prime **20** | Il modello impara; gli **iperparametri** si ottimizzano qui con walk-forward interno. |
| **Validation** | successive **5** | **Scelta del modello finale**: quali feature, quale algoritmo, calibrazione. Si "vedono" solo per decidere. |
| **Testing** | ultime **5** | **Intoccabili** fino alla fine. Giudizio onesto e definitivo sulla tesi: si guardano **una sola volta**. |

**Walk-forward espandente** (la tua proposta, applicata dentro questo schema):
1. **Train iniziale:** le **20 stagioni** (regular season **+ relativi playoff**).
2. **Prima stagione di validation:** inseriamo la regular season → il modello **predice** il bracket.
3. Arrivano i **risultati reali** dei playoff → **metriche** e poi **ri-allenamento** includendo quella stagione.
4. Si ripete espandendo la finestra, una stagione alla volta, attraverso il blocco validation; le **ultime 5** restano come **test finale mai visto**.

> **Nota metodologica critica (anti-overfitting):** la **ri-ottimizzazione degli iperparametri** si fa con **validazione interna alla finestra di training** (es. le ultime stagioni del train come validation interna), **non** guardando il risultato futuro che stiamo predicendo. I risultati reali servono ad **aggiungere dati** (ri-allenare), non a "sbirciare". Il blocco **testing (ultime 5)** non influenza **nessuna** scelta: si tocca solo per il verdetto finale. Questo protegge la validità della tesi.

### 5.4 Metriche
- Per serie: accuracy, **log-loss**, **Brier score**, calibrazione.
- Per bracket: % serie corrette, % round corretti, distanza dal bracket reale.
- Per stadio/titolo: log-loss sulle probabilità di campione.
- **Baseline di confronto:** seed (la testa di serie più alta vince), record W-L, net rating, mercato (odds se reperibili) → dobbiamo battere questi.

### 5.5 Test della tesi (§0 B/C)
- **Feature importance + SHAP** globali e per le 5 feature-tesi.
- **Ablation:** modello completo vs modello senza T1–T5 (quanto perdiamo?).
- **Modello solo-tesi (T1–T5)** vs completo (quanto spiega la tesi da sola?).
- Analisi per epoca (la tesi regge nel tempo? il tiro da 3 conta di più dopo il 2015?).

---

## 6. Fasi di lavoro (incrementali)

- **Fase 0 — Audit copertura dati**: quanti giocatori hanno Combine? Le 30 stagioni hanno tutte stat affidabili? Verifica fattibilità clustering e completezza head-to-head/serie. *(Gate decisionale: se una stagione antica risultasse troppo lacunosa, si segnala prima di costruirci sopra.)*
- **Fase 1 — Raccolta dati + caching**: nba_api + B-Ref, salvataggio locale, dataset grezzi giocatore/squadra/serie.
- **Fase 2 — Feature engineering**: blocchi giocatore (§3), squadra (§4), i due clustering, aging curve.
- **Fase 3 — Modello base + walk-forward** (§5).
- **Fase 4 — Bracket Monte Carlo + backtest storico**.
- **Fase 5 — Test della tesi** (ablation, SHAP, solo-tesi).
- **Fase 6 — Visualizzazioni** (§7).

> Decideremo insieme fin dove spingere; le fasi sono indipendenti abbastanza da fermarsi a una "fetta verticale" funzionante prima.

---

## 7. Visualizzazioni (per dimostrare la tesi)

- **Bracket predetto vs reale** (per ogni stagione di test), con probabilità.
- **Importanza delle feature-tesi** (SHAP summary, dependence plots per T1–T5).
- **Hometown vs successo:** scatter/serie storica share-hometown vs profondità playoff raggiunta.
- **Tiro da 3 nel tempo** e correlazione con vittorie (evoluzione per epoca).
- **Distribuzione del contributo** (T5): campioni vs eliminati al 1° turno.
- **Matrice matchup** (heatmap head-to-head) e casi "upset spiegati dal matchup".
- **Calibrazione** del modello (reliability diagram).
- **Profili/archetipi** dei roster vincenti (radar chart).

---

## 8. Rischi e limiti (onestà intellettuale)

- **Causalità:** da dati osservazionali possiamo mostrare **associazione forte e potere predittivo**, non causalità pura. Ablation e controlli avvicinano, ma il claim resta "questi fattori predicono/spiegano", non "causano" in senso sperimentale.
- **Sample size:** ~16 campioni di playoff a 30 anni sono **pochi** per il titolo finale → puntare sulle **serie** (centinaia) come unità statistica, non sui titoli.
- **Combine mancante:** rischio principale; il proxy via clustering mitiga ma introduce rumore (tracciato con `is_imputed`).
- **Rate limit / fragilità scraping:** mitigati con caching aggressivo e retry.
- **Regole cambiate nel tempo** (hand-check, pace, importanza del 3): da gestire con normalizzazioni per epoca / per-100-possessi.
- **Infortuni & roster di playoff:** un titolo può saltare per un infortunio — rumore irriducibile; lo modelliamo come incertezza (Monte Carlo).

---

## 9. Stack tecnico & struttura

- **Linguaggio:** Python.
- **Formato consegna:** **moduli `.py` (pipeline)** + **notebook Jupyter** (esplorazione e visualizzazioni).
- **Struttura proposta:**

```
nba-playoff-predictor/
├── PROJECT_SPEC.md            # questo documento
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                   # cache chiamate API / scraping
│   ├── interim/
│   └── processed/             # dataset finali (parquet)
├── src/
│   ├── collect/               # nba_api + basketball-reference + caching
│   ├── features/              # player, team, clustering, aging
│   ├── model/                 # serie, bracket MC, walk-forward
│   ├── evaluate/              # metriche, backtest, ablation/SHAP
│   └── viz/                   # grafici
├── notebooks/
│   ├── 00_data_audit.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_model_walkforward.ipynb
│   ├── 04_thesis_tests.ipynb
│   └── 05_visualizations.ipynb
└── config.yaml                # orizzonte anni, soglie hometown, ecc.
```

- **Librerie principali:** `nba_api`, `pandas`, `numpy`, `scikit-learn`, `xgboost`/`lightgbm`, `shap`, `matplotlib`/`seaborn`/`plotly`, `requests`+`beautifulsoup4` (B-Ref), `pyarrow`.
- **Ambiente:** virtualenv dedicato (coerente con i tuoi altri progetti).

---

## 10. Decisioni confermate (riepilogo)

| Tema | Decisione |
|------|-----------|
| Target | Modello modulare: serie → bracket → probabilità stadio → campione |
| Atletismo mancante | Clustering + proxy |
| Fonte dati | nba_api primario + Basketball-Reference per i buchi |
| Validazione | Split temporale **20 train / 5 validation / 5 test** + walk-forward espandente |
| Stat playoff | Sì playoff: come storico di rendimento **e** per accorciamento rotazioni (anti-leakage) |
| Test tesi | Sia come feature **sia** analisi causale dedicata (ablation/SHAP/solo-tesi) |
| Hometown | Scala graduata (0–4) |
| Clustering livelli | Sì: superstar/all-star/starter/role/bench/bench-warmer |
| Formato | Moduli `.py` + notebook Jupyter |
| Orizzonte | **30 anni** (20 train / 5 validation / 5 test); copertura verificata in Fase 0 |

---

## 11. Cosa serve da te per partire

1. **Approvazione** di questo documento (o correzioni).
2. Conferma del **punto di partenza**: propongo di iniziare dalla **Fase 0 (audit copertura dati)** — è veloce e ci dice se 30 anni sono realistici prima di investire nel resto.

> Una volta approvato, trasformo §6 in una todo-list operativa e parto dalla Fase 0.
