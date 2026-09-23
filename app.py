"""Differential Evolution - Mutation über den Differenzvektor der Population - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Populations-Metaheuristiken-Linie der "Konzepte"-Reihe, ein weiterer KONTRAST zu GA (kein Fix) für
kontinuierliche Landschaften: Differential Evolution (Storn & Price, 1997) ersetzt GAs Crossover/Mutation durch einen
viel einfacheren Mechanismus als CMA-ES - die Mutation ist ein skalierter Differenzvektor zweier zufälliger
Populationsmitglieder, die Streuung der Population selbst liefert das Schrittweiten-Signal. Vehikel ist dieselbe
kontinuierliche Standortwahl wie genetic-algorithm-demo/cma-es-demo - direkt vergleichbar mit deren gemessenen
Befunden auf derselben Landschaft.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import de_constants as C
from de_evaluation import Settings, analyse, comparison_experiment, f_experiment, sweep
from de_presets import apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_run_seed, randomize_seed, sync_query_params
from de_visualization import build_best_curve, build_comparison, build_diversity_curve, build_f_experiment, build_growing_example, build_sweep

st.set_page_config(page_title="Differential Evolution – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings, keep_history=True)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _comparison():
    return comparison_experiment()


@st.cache_data(show_spinner=False)
def _f_experiment():
    return f_experiment()


st.title("🧬 Differential Evolution – Mutation über den Differenzvektor der Population")
st.markdown(
    """
GA kombiniert Individuen per Crossover, CMA-ES verfolgt eine einzige sich entwickelnde Gauß-Verteilung. **Differential
Evolution** (Storn & Price, 1997) geht für kontinuierliche Landschaften einen dritten, sehr einfachen Weg: die Mutation
ist ein **skalierter Differenzvektor** zweier zufälliger Populationsmitglieder, addiert auf ein drittes -
$v_i = x_{r_1} + F \\cdot (x_{r_2} - x_{r_3})$. Die aktuelle Streuung der Population liefert damit selbst das
Schrittweiten-Signal, ganz ohne gelernte Kovarianzmatrix. Eine **binomiale Rekombination** mischt Mutant und Original,
eine **gierige Ein-zu-eins-Selektion** ersetzt ein Individuum nur, wenn das Kind mindestens gleich gut ist.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - "
    "sechstes Stück der Populations-Metaheuristiken-Linie der \"Konzepte\"-Reihe, ein **Kontrast** zu "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) statt eines Fixes - "
    "**ein** Verfahren an einem wachsenden Beispiel. Vehikel ist dieselbe kontinuierliche Standortwahl wie dort/bei "
    "[cma-es-demo](https://sebastianhanisch-cma-es-demo.streamlit.app/)."
)

with st.expander("So funktioniert Differential Evolution", expanded=True):
    st.markdown(
        r"""
1. **Mutation.** Für jedes Individuum $i$: drei verschiedene, von $i$ verschiedene Populationsmitglieder $x_{r_1},
   x_{r_2}, x_{r_3}$, Mutant $v_i = x_{r_1} + F \cdot (x_{r_2} - x_{r_3})$ - das Differenzgewicht $F$ skaliert den
   Schritt.
2. **Rekombination.** Binomiale Kreuzung: jede Dimension des Kindes stammt mit Wahrscheinlichkeit $CR$ vom Mutanten,
   sonst vom Original - mindestens eine Dimension kommt garantiert vom Mutanten.
3. **Selektion.** Das Kind ersetzt das Original nur, wenn es mindestens gleich gut ist (gierig, Ein-zu-eins) -
   die Population wird nie schlechter.
4. **Selbstadaption ohne Kovarianzmatrix.** Die Schrittweite ergibt sich implizit aus der aktuellen Streuung der
   Population selbst: schrumpft die Population um ein Optimum, werden auch die Mutationsschritte automatisch kleiner.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
cols = st.columns(len(preset_names))
for col, name in zip(cols, preset_names):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Differential Evolution**")
    pop = st.slider("Populationsgröße NP", *bounds("pop_slider"), key="pop_slider", step=C.POP_STEP)
    generations = st.slider("Generationen", *bounds("gens_slider"), key="gens_slider", step=C.GEN_STEP)
    f = st.slider("Differenzgewicht F", *bounds("f_slider"), key="f_slider", step=C.F_STEP, help="Skaliert den Mutations-Schritt (Differenzvektor zweier zufälliger Individuen).")
    cr = st.slider("Crossover-Rate CR", *bounds("cr_slider"), key="cr_slider", step=C.CR_STEP, help="Anteil der Dimensionen, die vom Mutanten statt vom Original übernommen werden.")
    seed = st.number_input("Zufalls-Seed des Vehikels", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neues Vehikel generieren", width="stretch", on_click=randomize_seed)
    run_seed = st.number_input("Zufalls-Seed des DE-Laufs", *bounds("run_seed_input"), key="run_seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_run_seed)

sync_query_params({
    "pop_slider": int(pop), "gens_slider": int(generations), "f_slider": float(f), "cr_slider": float(cr),
    "seed_input": int(seed), "run_seed_input": int(run_seed),
})

settings = Settings(seed=int(seed), pop=int(pop), gens=int(generations), f=float(f), cr=float(cr), run_seed=int(run_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
result = a.result
n_gens_run = len(result.generations) - 1
data_key = settings

# --- DE in Aktion ----------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 DE in Aktion")
if "de_gen" not in st.session_state or st.session_state.get("de_gen_owner") != data_key:
    st.session_state["de_gen"] = n_gens_run
    st.session_state["de_gen_owner"] = data_key
gen_col, play_col = st.columns([5, 2])
with gen_col:
    gen = st.slider("Generation", 0, n_gens_run, key="de_gen", help="0 = Startpopulation.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_gens_run == 0:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_gens_run, min(n_gens_run + 1, 40))})


def _render(g):
    gd = result.generations[g]
    with view_slot.container():
        c1, c2 = st.columns([3, 2])
        c1.markdown(f"**Generation {g} von {n_gens_run} – Diversität: {result.diversity_history[g]:.3f}**")
        c1.plotly_chart(build_growing_example(a.inst, gd, a.grid_xy), width="stretch", key=f"g_map_{g}")
        c2.markdown("**Bester Fund bisher**")
        c2.plotly_chart(build_best_curve(result.best_history[:g + 1], reference=a.grid_cost), width="stretch", key=f"g_best_{g}")


if auto_play:
    for fr in _frames():
        _render(fr)
        time.sleep(0.15)
else:
    _render(gen)

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was DE gefunden hat")
m1, m2, m3 = st.columns(3)
m1.metric("Im globalen Trichter gelandet?", "Ja" if a.found_global else "Nein")
m2.metric("Abstand zum Gitter-Optimum", f"{a.gap:+.1f} %")
m3.metric("Diversität am Ende", f"{result.diversity_history[-1]:.4f}", delta=f"Start {result.diversity_history[0]:.2f}", delta_color="off")
st.plotly_chart(build_diversity_curve(result.diversity_history), width="stretch", key="diversity_curve")

st.markdown("---")

# --- Sweep -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt die Trefferquote von Populationsgröße und Differenzgewicht ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(C.SWEEP_LABELS), format_func=lambda k: C.SWEEP_LABELS[k], key="sweep_select")
base_sweep = Settings(seed=settings.seed, gens=settings.gens, pop=settings.pop, f=settings.f, cr=settings.cr)
if st.button("Sweep über 5 feste Vehikel berechnen (dauert etwa 10 bis 30 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, C.SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")

st.markdown("---")

# --- Experiment 1: Kopfexperiment gegen GA und CMA-ES -----------------------------------------------------------------------------------------

st.subheader("🔬 Wie schlägt sich DE gegen GA und CMA-ES?")
st.caption(
    f"Dieselbe Standortwahl-Landschaft wie genetic-algorithm-demo/cma-es-demo (Vehikel-Seed {C.DEFAULT_SEED}) - dort trafen "
    f"GA {C.GA_SUCCESS_SMALL:.0%}/{C.GA_SUCCESS_LARGE:.0%} und CMA-ES {C.CMA_SUCCESS_SMALL:.0%}/{C.CMA_SUCCESS_LARGE:.0%} "
    f"die globale Mulde bei denselben zwei Budgets. DE bekommt dieselben Budgets, Standard-NP/F/CR - Ausgang vorab offen."
)
if st.button("DE gegen GA und CMA-ES rechnen (dauert etwa 10 Sekunden)", key="comparison_start"):
    st.session_state["comparison_on"] = True
if st.session_state.get("comparison_on"):
    with st.spinner("Rechne 20 DE-Läufe je Budget..."):
        report = _comparison()
    st.plotly_chart(build_comparison(report), width="stretch", key="comparison_chart")
    c1, c2 = st.columns(2)
    c1.metric(f"DE, {report['de_small_evals']} Auswertungen", f"{report['de_small']:.0%}", delta=f"GA {report['ga_small']:.0%} · CMA-ES {report['cma_small']:.0%}", delta_color="off")
    c2.metric(f"DE, {report['de_large_evals']} Auswertungen", f"{report['de_large']:.0%}", delta=f"GA {report['ga_large']:.0%} · CMA-ES {report['cma_large']:.0%}", delta_color="off")
    st.warning(
        "**Ehrlicher Befund:** Mit Standardeinstellungen trifft DE hier bei BEIDEN Budgets zuverlässig die globale Mulde - "
        "deutlich robuster als CMA-ES (das nur eine einzige Gauß-Glocke verfolgt) und auch klar vor GA. Der Grund ist "
        "strukturell: wie GA hält DE eine über den Raum verstreute Population, jedes Individuum hat eine eigene Chance "
        "auf die richtige Mulde - und die Differenzvektor-Mutation nutzt die Streuung selbst als Schrittweiten-Signal, "
        "ohne dass eine einzelne Verteilung vorzeitig kollabieren kann. Das ist kein Beweis, dass DE grundsätzlich besser "
        "ist (die Landschaft mit fünf gut getrennten Mulden und einer für DE komfortablen Standard-Populationsgröße "
        "begünstigt diesen Mechanismus) - der Effekt eines knapperen Budgets zeigt sich stattdessen im Regler-Experiment "
        "unten und im Sweep."
    )

st.markdown("---")

# --- Experiment 2: eigener Regler - Differenzgewicht F ----------------------------------------------------------------------------------------

st.subheader("🔬 Wie stark hängt die Trefferquote vom Differenzgewicht F ab?")
st.caption(f"Eigenes knappes Budget ({C.F_EXPERIMENT_POP} Individuen, {C.F_EXPERIMENT_GENS} Generationen) - beim komfortablen Standardfall sättigt die Trefferquote fast überall bei 100 %.")
if st.button(f"Differenzgewichte {C.F_VALUES[0]:.1f} bis {C.F_VALUES[-1]:.1f} vergleichen (dauert etwa 5 Sekunden)", key="f_start"):
    st.session_state["f_on"] = True
if st.session_state.get("f_on"):
    with st.spinner("Rechne 5 Differenzgewichte × 10 Läufe..."):
        rows_f = _f_experiment()
    st.plotly_chart(build_f_experiment(rows_f), width="stretch", key="f_chart")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Individuen für eine sinnvolle Streuung** | Bei sehr kleinem NP (nahe der strukturellen Untergrenze von 4) liefert der Differenzvektor kaum brauchbares Richtungssignal - die Trefferquote sinkt (gemessen im Sweep). | Größere Populationsgröße NP |
| **Differenzgewicht F ist gut gewählt** | Zu klein: Mutation zu schwach, Population konvergiert vorzeitig. Bei knappem Budget gemessen: 20 % Trefferquote bei F=0,1 gegen 90 % bei F=0,8-1,5. | Muss von Hand eingestellt werden, wie bei jedem Regler dieser Linie |
| **Box-Constraints passen zur Landschaft** | Diese Demo begrenzt die Population per Clipping auf [0, 100]² - anders als cma-es-demos bewusst unbeschränkter Suchraum. Ohne Grenzen könnte die Mutation beliebig weit hinauslaufen. | Bewusste Vereinfachung, siehe README |
| **Kein Restart/Archiv bei Stagnation** | Bleibt die Population einmal in einer Mulde hängen, gibt es (anders als bei MOEA/Ds externem Archiv) keinen Mechanismus, das zu erkennen oder zu korrigieren. | Restart-Strategien, Diversitäts-Erhaltungsmechanismen - hier nicht umgesetzt |
"""
)
st.caption(
    "DE ist ein Geschwister von CMA-ES und Partikelschwarm-Optimierung (beide ebenfalls Kontrast-Kinder von GA für "
    "kontinuierliche Landschaften) - der geplante Fix-Nachfolger L-SHADE (selbstadaptives F/CR) ist ein separates, "
    "noch nicht gebautes Stück. Vorgänger: "
    "[genetic-algorithm-demo](https://sebastianhanisch-genetic-algorithm-demo.streamlit.app/) und "
    "[cma-es-demo](https://sebastianhanisch-cma-es-demo.streamlit.app/), deren Befunde hier direkt verglichen werden."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Mutation.** $v_i = x_{r_1} + F \cdot (x_{r_2} - x_{r_3})$, mit $r_1, r_2, r_3$ paarweise verschiedenen, von $i$
verschiedenen Indizes.

**Rekombination (binomial).** Für jede Dimension $j$: $u_{i,j} = v_{i,j}$ falls $\text{rand}_j < CR$ oder $j = j_{rand}$,
sonst $u_{i,j} = x_{i,j}$ - $j_{rand}$ garantiert mindestens eine Dimension vom Mutanten.

**Selektion.** $x_i \leftarrow u_i$ falls $f(u_i) \le f(x_i)$, sonst bleibt $x_i$ unverändert (gierig, Ein-zu-eins).

Implementiert in `de_algorithm.py` (Mutation, Rekombination, Hauptschleife), `de_scenario.py` (Vehikel),
`de_evaluation.py` (Kennzahlen, Sweep, Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
