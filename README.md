# 🧬 Differential Evolution – Mutation über den Differenzvektor der Population

**[→ Demo live ausprobieren](https://sebastianhanisch-differential-evolution-demo.streamlit.app/)**

Sechstes Stück der **Populations-Metaheuristiken-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) –
Operations Research und Machine Learning. Zweiter Kontrast zu [genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/)
für kontinuierliche Landschaften, Geschwister von [cma-es-demo](https://sebastianhanisch-cma-es-demo.streamlit.app/):
Differential Evolution (Storn & Price, 1997) erreicht Selbstadaption der Schrittweite über einen viel einfacheren
Mechanismus als CMA-ES - die Mutation ist ein skalierter Differenzvektor zweier zufälliger Populationsmitglieder, die
Streuung der Population selbst liefert das Schrittweiten-Signal. Vehikel ist dieselbe kontinuierliche Standortwahl wie
genetic-algorithm-demo/cma-es-demo - direkt vergleichbar mit deren gemessenen Befunden auf derselben Landschaft.

## Warum dieses Problem

CMA-ES lernt eine explizite Kovarianzmatrix, um die Form der Landschaft nachzubilden - mächtig, aber mit spürbarem
Buchhaltungsaufwand (Eigenzerlegung, zwei Evolutionspfade, mehrere Adaptionsraten). DE erreicht einen ähnlichen Effekt
(die Schrittweite schrumpft automatisch, wenn die Population konvergiert) mit einem viel einfacheren Mechanismus:
**kein** gelerntes Modell der Landschaft, sondern die aktuelle **Streuung der Population selbst** als Signal. Die
Mutation nimmt zwei zufällige Individuen, bildet ihren Differenzvektor und skaliert ihn mit einem festen Faktor $F$ -
ist die Population noch weit verstreut, sind die Schritte groß; hat sie sich um ein Optimum verdichtet, werden sie
automatisch klein.

## Modell

Dieselbe kontinuierliche Standortwahl wie genetic-algorithm-demo/cma-es-demo: Kosten eines Punkts $(x,y)$ im
100×100-km-Gebiet sind mehrere Gauß-Mulden unterschiedlicher Tiefe/Breite (`K_WELLS=5`) - nur die tiefste ist das
globale Optimum. **`de_scenario.generate_real` reproduziert die Vehikel-Erzeugung wortgleich** - bei
Standard-Vehikel-Seed 35 bitidentisch zu genetic-algorithm-demo/cma-es-demo, direkt zitierbare Vergleichszahlen. Die
Population wird per **Box-Constraint** auf [0, 100]² begrenzt (Clipping nach jeder Mutation/jedem Crossover) - anders
als cma-es-demos bewusst unbeschränkter Suchraum, weil DEs Standardmechanik (Storn & Price) auf begrenzte Suchräume
ausgelegt ist und GAs eigene reelle Kodierung ebenso Grenzen nutzt.

## Methodik

Klassisches DE/rand/1/bin (`de_algorithm.py`, kein GA-/CMA-ES-Kern kopiert - andere Mechanik): pro Individuum $i$ ein
Mutant $v_i = x_{r_1} + F \cdot (x_{r_2} - x_{r_3})$ aus drei zufälligen, von $i$ verschiedenen Populationsmitgliedern,
binomiale Rekombination mit dem Original (Crossover-Rate $CR$, mindestens eine Dimension garantiert vom Mutanten),
gierige Ein-zu-eins-Selektion (das Kind ersetzt das Original nur bei mindestens gleich gutem Wert - die Population
wird nie schlechter).

**Kreuzprobe gegen `scipy.optimize.differential_evolution`** (Referenzimplementierung, `strategy="rand1bin"` passend
zur eigenen DE/rand/1/bin): Mutation/Crossover per Handrechnung an einer konstruierten Population geprüft; beide
Implementierungen finden auf einfachen konvexen Testfunktionen (Kugel-, Ellipsoid-Funktion) mit vergleichbarem Budget
verlässlich nahe an 0 - kein Generation-für-Generation-Gleichlauf behauptet (unterschiedliche RNG-Nutzung).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Wie schlägt sich DE gegen GA und CMA-ES auf derselben mehrgipfligen Landschaft? | Mit Standardeinstellungen trifft DE die globale Mulde in **100 %** der Läufe - bei BEIDEN Budgets (1520 und 15120 Auswertungen). GA erreicht dort 55 %/95 %, CMA-ES nur 15 %/15 %. | `test_comparison_experiment_headline_claims` |
| Warum ist DE hier so robust? | DE hält wie GA eine über den Raum verstreute Population - jedes Individuum hat eine eigene Chance auf die richtige Mulde. Die Differenzvektor-Mutation nutzt die Streuung selbst als Schrittweiten-Signal, ohne dass eine einzelne Verteilung (wie bei CMA-ES) vorzeitig kollabieren kann. | mechanistisch, im Code nachvollziehbar |
| Ist das ein Beweis, dass DE grundsätzlich besser ist? | Nein - die Landschaft mit fünf gut getrennten Mulden und einer für DE komfortablen Standard-Populationsgröße begünstigt diesen Mechanismus. Bei knapperem Budget zeigt sich das Differenzgewicht F sehr wohl als Stellschraube. | `test_f_experiment_headline_claims` |
| Wie stark hängt die Trefferquote vom Differenzgewicht F ab (knappes Budget)? | Klar: von 20 % (F=0,1) auf 90 % (F≥0,8) über 10 Seeds - zu schwache Mutation lässt die Population vorzeitig konvergieren. | `test_f_experiment_headline_claims` |
| Stimmen Mutation/Crossover mit der Literatur überein? | Per Handrechnung geprüft; beide Implementierungen (eigene und `scipy`) konvergieren auf Kugel-/Ellipsoid-Funktionen verlässlich nahe an 0 | `test_run_de_and_scipy_reach_a_comparably_good_optimum_on_the_sphere` |

## Ehrliche Grenzen

- **Kein Beweis genereller Überlegenheit** - der klare Vorsprung gegenüber GA und CMA-ES gilt für DIESE Landschaft
  (fünf gut getrennte Mulden, komfortable Standard-Populationsgröße). Andere Landschaften (z. B. sehr hochdimensional,
  stark korrelierte Variablen) könnten CMA-ES' gelernte Kovarianzmatrix wieder in Vorteil bringen - hier nicht geprüft.
- **Differenzgewicht F ist trotzdem ein echter Regler** - bei knappem Budget zeigt sich der erwartete Effekt klar
  (20 % bis 90 % Trefferquote), auch wenn er beim komfortablen Standardbudget durch Sättigung überdeckt wird.
- **Box-Constraints per Clipping** - eine einfache, aber nicht die einzige Möglichkeit, DE auf einen begrenzten
  Suchraum zu beschränken (Alternativen: Reflexion, Neuziehen). Hier bewusst die einfachste Variante gewählt.
- **Kein Restart/Archiv bei Stagnation** - bleibt die Population einmal in einer Mulde hängen, gibt es (anders als
  MOEA/Ds externes Archiv) keinen Mechanismus, das zu erkennen oder zu korrigieren.
- **Struktureller Mindestbedarf NP≥4** - die Mutation braucht drei verschiedene, vom Ziel-Individuum verschiedene
  Populationsmitglieder.
- **Kein Nachfolger in dieser Demo geplant** - der geplante Fix-Nachfolger L-SHADE (selbstadaptives F/CR) ist ein
  separates, noch nicht gebautes Stück. DE ist außerdem ein Geschwister von CMA-ES und Partikelschwarm-Optimierung
  (beide ebenfalls Kontrast-Kinder von GA für kontinuierliche Landschaften).

## Tests

58 Tests (`pytest tests/ -v`): Mutation/Crossover per Handrechnung an einer konstruierten Population geprüft,
Konvergenz auf Kugel-/Ellipsoid-Funktion (eigene Implementierung UND `scipy.optimize.differential_evolution` im
Vergleich), Szenario-Erzeugung bitidentisch zu genetic-algorithm-demo/cma-es-demo geprüft, AppTest-Rauchtests (jedes
Preset, Generation-Slider inkl. Abspielen, Permalink-Grenzen, beide Experimente + Sweep auf Abruf) und
`test_claims.py` (jede Zahl aus diesem README, mit CI-robusten Bändern für Einzellauf-Kennzahlen - siehe
`feedback_ci_platform_robust_tests.md`, von Anfang an angewendet).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `de_constants.py` | Regler-Grenzen, Vehikel-Konstanten, Presets |
| `de_presets.py` | Permalink/Presets-Mechanik |
| `de_scenario.py` | Vehikel-Erzeuger (Standortwahl, Gauß-Mulden-Landschaft), wortgleich zu genetic-algorithm-demo/cma-es-demo |
| `de_algorithm.py` | DE-Kern (Mutation, Crossover, Hauptschleife) |
| `de_evaluation.py` | Kennzahlen, Kopfexperiment, Differenzgewicht-Experiment, Sweep |
| `de_visualization.py` | Plotly-Abbildungen (Landschaft mit Population, Vergleiche) |

## Bewusst nicht umgesetzt

- L-SHADE (selbstadaptives F/CR) - geplanter Fix-Nachfolger, separates Stück.
- Alternative Bounds-Handhabung (Reflexion, Neuziehen statt Clipping).
- Restart-Strategien oder ein externes Archiv bei Stagnation.
- Ein PDF-Export - wie bei den anderen Konzepte-Demos dieses Portfolios nicht Teil der Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.
