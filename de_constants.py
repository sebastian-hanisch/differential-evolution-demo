"""Konstanten der DE-Demo: Vehikel (wie genetic-algorithm-demo/cma-es-demo, kontinuierliche Standortwahl),
DE-Regler, Presets (Presets folgen nach den Messungen)."""

# --- Vehikel: Standortwahl (wortgleich aus genetic-algorithm-demo/ga_constants.py) -------------------------------------------------------

AREA = 100.0
K_WELLS = 5
WELL_MARGIN = 12.0                # Schwerpunkte liegen mindestens so weit vom Rand entfernt
AMP_MIN, AMP_MAX = 15.0, 40.0     # Tiefe eines Trichters
SIGMA_MIN, SIGMA_MAX = 6.0, 14.0  # Breite eines Trichters
GRID_STEP = 1.0                   # Auflösung des Referenz-Gitters (Grid-Search-Minimum) in km

DIM = 2
BOUNDS = (0.0, AREA)              # Box-Constraint (Clipping) - anders als cma-es-demos bewusst unbeschränkter Suchraum,
                                   # siehe README: DEs Standardmechanik ist auf begrenzte Suchräume ausgelegt

# --- Differential Evolution ------------------------------------------------------------------------------------------------------------------

POP_MIN, POP_MAX, DEFAULT_POP, POP_STEP = 6, 200, 20, 2
GEN_MIN, GEN_MAX, DEFAULT_GEN, GEN_STEP = 10, 400, 100, 10
F_MIN, F_MAX, DEFAULT_F, F_STEP = 0.1, 2.0, 0.8, 0.05     # Differenzgewicht (Storn & Price übliche Spanne 0,4-1,0)
CR_MIN, CR_MAX, DEFAULT_CR, CR_STEP = 0.0, 1.0, 0.9, 0.05  # Crossover-Rate (binomiale Rekombination)
SEED_MAX = 999999
DEFAULT_SEED = 35                 # Vehikel-Seed (wie genetic-algorithm-demo/cma-es-demo, bitidentische Landschaft)
DEFAULT_RUN_SEED = 7              # Seed des DE-Laufs selbst (Population, Mutation, Crossover)

GLOBAL_TOL_KM = 3.0                # Standort gilt als "im globalen Trichter" gefunden, wenn er höchstens so weit vom besten Gitterpunkt entfernt liegt (wie GA/CMA-ES)

# --- Kopfexperiment: DE gegen die in genetic-algorithm-demo/cma-es-demo gemessenen Zahlen ---------------------------------------------------
# GA maß auf DERSELBEN Landschaft (Vehikel-Seed 35): Populationsgröße 10 (150 Generationen, 1510 Auswertungen) trifft
# die globale Mulde in 55 % von 20 Läufen, Populationsgröße 100 (15100 Auswertungen) in 95 %. cma-es-demo maß mit
# Standard-σ0/λ auf denselben zwei Budgets 15 % bei BEIDEN.

GA_SUCCESS_SMALL, GA_EVALS_SMALL = 0.55, 1510
GA_SUCCESS_LARGE, GA_EVALS_LARGE = 0.95, 15100
CMA_SUCCESS_SMALL, CMA_SUCCESS_LARGE = 0.15, 0.15
COMPARISON_SEEDS = tuple(range(1600000, 1600020))

# --- Eigener Regler: Differenzgewicht F -------------------------------------------------------------------------------------------------------

F_VALUES = (0.1, 0.3, 0.5, 0.8, 1.5)
F_EXPERIMENT_SEEDS = tuple(range(1700000, 1700010))
# eigenes, knappes Budget (unabhängig von der Seitenleiste, wie genetic-algorithm-demos OPERATOR_POP/OPERATOR_GENS) -
# beim komfortablen Standardfall (NP=20) sättigt die Trefferquote fast überall bei 100 %, der Effekt von F zeigt sich
# erst bei knapperem Budget deutlich
F_EXPERIMENT_POP = 10
F_EXPERIMENT_GENS = 40

SWEEP_SEEDS = tuple(range(1800000, 1800005))
SWEEP_VALUES = {"pop": (6, 10, 20, 40, 80), "f": F_VALUES}
SWEEP_LABELS = {"pop": "Populationsgröße NP", "f": "Differenzgewicht F"}


def _preset(pop=DEFAULT_POP, gens=DEFAULT_GEN, f=DEFAULT_F, cr=DEFAULT_CR, seed=DEFAULT_SEED, run_seed=DEFAULT_RUN_SEED):
    return {"pop": pop, "gens": gens, "f": f, "cr": cr, "seed": seed, "run_seed": run_seed}


PRESETS = {
    "Standardfall": _preset(),
    "Kleines Differenzgewicht": _preset(f=F_VALUES[0]),
    "Großes Differenzgewicht": _preset(f=F_VALUES[-1]),
    "Große Population": _preset(pop=80),
}
PRESET_HELP = {
    "Standardfall": "NP=20, F=0,8, 100 Generationen (Standard-Seed): trifft die globale Mulde fast exakt (-0,02 % Abstand), Diversität kollabiert auf 0.",
    "Kleines Differenzgewicht": "F=0,1 statt 0,8: trifft hier trotzdem die globale Mulde (0,01 % Abstand) - bei diesem großzügigen Budget (100 Generationen) sättigt fast jede Einstellung; der Unterschied zeigt sich erst beim knappen Budget im Regler-Experiment (20 % Trefferquote bei F=0,1 gegen 90 % bei F≥0,8).",
    "Großes Differenzgewicht": "F=1,5 statt 0,8: trifft ebenfalls die globale Mulde, bleibt am Ende aber deutlich vielfältiger (Diversität 0,005 statt 0) - die größeren Mutationsschritte lassen die Population nie ganz kollabieren.",
    "Große Population": "NP=80 statt 20, sonst wie im Standardfall: trifft dieselbe Mulde ebenso zuverlässig, aber mit dem 4-fachen Rechenaufwand je Generation - bei dieser Landschaft bringt die größere Population hier keinen zusätzlichen Nutzen (siehe Sweep).",
}
