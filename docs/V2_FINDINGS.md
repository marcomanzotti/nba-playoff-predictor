# V2 — Risultati della seconda iterazione (verdetti onesti)

> Questa seconda versione del progetto si e' posta due domande precise e ha
> ottenuto due risposte **oneste e controintuitive**. Il valore non sta in un
> modello "che vince", ma nell'aver stabilito con rigore *quanto in la' si puo'
> davvero spingere la predizione* e *cosa conta davvero* per vincere un titolo.

---

## Domanda 1 — Qual e' la RICETTA per massimizzare le chance di titolo?

### Come l'abbiamo affrontata (anti-tautologia)
La V1 partiva dalle 5 ipotesi T1–T5 e mostrava che "feature dei giocatori
spiegano il 62% del net rating". Critica giusta: dire *"servono giocatori forti"*
e' una tautologia, non una ricetta.

Quindi in V2 abbiamo:
1. separato le feature in **TALENTO** (livello/impatto del giocatore — un *esito*,
   9 feature) e **STRUTTURA** (caratteristiche costruibili e oggettive — fisico,
   tiro, esperienza, composizione per ruolo, pace — 37 feature);
2. cercato la ricetta su **tutte** le 46 feature, con selezione **out-of-time**
   (allena sul passato, misura sul futuro) per non gonfiare l'R^2;
3. costruito un target non-tautologico, l'**OVER-PERFORMANCE**:
   `deep_run_reale - deep_run_atteso_dal_solo_talento`. Cioe': *a parita' di
   talento*, cosa fa andare una squadra piu' lontano del previsto?

Codice: `src/evaluate/recipe_search.py`, `src/evaluate/title_recipe.py`,
`src/evaluate/over_performance.py`. Output: `data/processed/recipe.json`,
`recipe_structural.json`.

### Il verdetto
| Modello | Cosa misura | R^2 out-of-time |
|---|---|---|
| Tutte le feature → net rating | forza della squadra | **0.67** |
| Solo STRUTTURA → deep-run | leve costruibili, ignorando il talento | **0.21** |
| A PARITA' di talento (over-perf) → deep-run | la ricetta non-tautologica | **0.03** |

**Conclusione onesta:**
- Il **talento spiega ~l'80%** di chi va lontano. Su questo non serviva la
  statistica: i giocatori forti vincono.
- **A parita' di talento**, l'unica leva strutturale ripetibile e' l'**esperienza
  playoff** (`T3_playoff_depth`, segno positivo). Tiro da 3, dimensione e
  atletismo **non** danno un vantaggio sistematico una volta tolto il talento.
- Tutto il resto (matchup, infortuni, fortuna) e' **varianza** irriducibile a
  questo livello di analisi (over-performance: media ~0, std 0.74 su scala 0–4).

> La ricetta del titolo, empiricamente: **prendi i giocatori piu' forti che puoi,
> poi privilegia chi ha gia' esperienza di playoff. Oltre questo, e' varianza.**
> Le 5 ipotesi originali (T1–T5) NON sono il driver dominante: lo e' il talento,
> e tra le ipotesi solo l'esperienza (T3) sopravvive al controllo del talento.

---

## Domanda 2 — Si puo' predire meglio CHI VINCE LA SERIE coi matchup di stile?

### Come l'abbiamo affrontata
Il modello-serie V1 vedeva solo *differenze* feature-per-feature (`d_<col>`), che
non catturano il "forte contro X, debole contro Y". Abbiamo aggiunto **7 feature
di interazione di stile ANTISIMMETRICHE** (offesa-vs-difesa, tiro-vs-difesa,
size mismatch, atletismo, pace clash, spacing-vs-size).

Codice: `src/features/series_interactions.py`, agganciate in
`src/model/series_dataset.py`. **Antisimmetria verificata numericamente**
(`max|base + mirror| = 0.0`, label balance = 0.500): la simmetria del modello e'
intatta. Tutte e 7 sopravvivono alla selezione per scorrelazione (informazione
genuina, non duplicati).

### Il verdetto
| Metrica (TEST 2021–2025) | V1 (senza matchup) | V2 (con matchup) |
|---|---|---|
| Accuracy | 65–68% | **65.3%** |
| Log loss | 0.636 | **0.636** |
| Brier | 0.222 | **0.221** |

**Le interazioni NON migliorano la predizione.** E, come in V1, il **baseline
"chi ha il record migliore" (win_pct) batte il modello completo** (log loss 0.558
vs 0.636).

**Perche' (onesto):**
1. XGBoost (alberi) costruisce gia' da solo le interazioni tra `d_OFF_RATING` e
   `d_DEF_RATING`: dargliele pre-confezionate non aggiunge nulla. (Le feature di
   interazione esplicite aiutano i modelli *lineari*, non gli alberi.)
2. 150 serie di test sono poche e rumorose: l'accuratezza per anno oscilla tra
   53% (2023/24, quasi coin-flip) e 93% (2021).
3. A questa scala il segnale di matchup, se esiste, e' piu' piccolo del rumore.

> **I playoff NBA sono dominati da "chi e' piu' forte" + caso.** Il margine
> predittivo oltre il record di regular season e' minuscolo e non
> statisticamente affidabile con 30 serie/anno. Le interazioni restano nel
> modello come **lente di analisi dei matchup** (mostrabili in dashboard), non
> come miglioramento predittivo.

---

## Perche' questi "non-risultati" sono il risultato

Un modello che batte di poco un baseline gonfiando feature sarebbe stato meno
onesto. Aver **stabilito con rigore il tetto della predicibilita'** (e che la
ricetta del titolo si riduce a talento + esperienza) e' una conclusione forte e
difendibile: distingue cio' che e' *costruibile e prevedibile* da cio' che e'
*varianza*. Questo e' esattamente cio' che la statistica deve fare — anche
quando la risposta e' "meno di quanto speravi".
