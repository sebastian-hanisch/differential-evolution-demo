"""Auswertung der DE-Demo: ein Lauf gegen das Gitter-Optimum, Sweep über NP/F, und zwei Experimente -
Kopfexperiment (Trefferquote im globalen Trichter gegen die in genetic-algorithm-demo/cma-es-demo gemessenen Zahlen)
und eigener Regler (Differenzgewicht F)."""

from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import de_algorithm as A
import de_constants as C
import de_scenario as S


@dataclass(frozen=True)
class Settings:
    seed: int = C.DEFAULT_SEED
    pop: int = C.DEFAULT_POP
    gens: int = C.DEFAULT_GEN
    f: float = C.DEFAULT_F
    cr: float = C.DEFAULT_CR
    run_seed: int = C.DEFAULT_RUN_SEED


@lru_cache(maxsize=64)
def instance(seed):
    inst = S.generate_real(seed)
    grid_xy, grid_cost = S.grid_optimum(inst)
    return inst, grid_xy, grid_cost


def cost_fn(settings):
    inst, grid_xy, grid_cost = instance(settings.seed)

    def cost(pop):
        return inst.cost(pop)
    return cost, grid_xy, grid_cost


def run(settings, keep_history=False):
    cost, grid_xy, grid_cost = cost_fn(settings)
    return A.run_de(cost, 2, C.BOUNDS, settings.pop, settings.gens, settings.f, settings.cr, settings.run_seed, keep_history=keep_history)


@dataclass
class Analysis:
    settings: Settings
    result: object
    inst: object
    grid_xy: np.ndarray
    grid_cost: float

    @property
    def gap(self):
        if self.grid_cost == 0:
            return float("nan")
        return 100.0 * (self.result.best_fitness - self.grid_cost) / abs(self.grid_cost)

    @property
    def found_global(self):
        return bool(np.sqrt(((self.result.best_individual - self.grid_xy) ** 2).sum()) <= C.GLOBAL_TOL_KM)


def analyse(settings, keep_history=True):
    result = run(settings, keep_history=keep_history)
    inst, grid_xy, grid_cost = instance(settings.seed)
    return Analysis(settings, result, inst, grid_xy, grid_cost)


# --- Sweep (wie die Vorgänger-Demos) ---------------------------------------------------------------------------------------------------------


def run_config(param, value, base, seeds=None):
    seeds = C.SWEEP_SEEDS if seeds is None else seeds
    s0 = replace(base, **{param: value})
    hits, gaps = [], []
    for run_seed in seeds:
        a = analyse(replace(s0, run_seed=run_seed), keep_history=False)
        hits.append(a.found_global)
        gaps.append(a.gap)
    return {"share_global": float(np.mean(hits)), "gap": float(np.mean(gaps))}


def sweep(param, base=None, values=None):
    base = Settings() if base is None else base
    values = C.SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(param, v, base)} for v in values]


# --- Experiment 1: Kopfexperiment gegen GA/CMA-ES-Zahlen -------------------------------------------------------------------------------------


def comparison_experiment(seed=None, seeds=None, pop=None, f=None, cr=None):
    """Trefferquote im globalen Trichter bei zwei Budgets (Gesamtauswertungen ~GA_EVALS_SMALL/LARGE), Standard-NP/F/CR -
    direkt vergleichbar mit den in genetic-algorithm-demo/cma-es-demo gemessenen Zahlen auf derselben Landschaft."""
    seed = C.DEFAULT_SEED if seed is None else seed
    seeds = C.COMPARISON_SEEDS if seeds is None else seeds
    pop = C.DEFAULT_POP if pop is None else pop
    f = C.DEFAULT_F if f is None else f
    cr = C.DEFAULT_CR if cr is None else cr

    rows = {}
    for label, evals in (("small", C.GA_EVALS_SMALL), ("large", C.GA_EVALS_LARGE)):
        gens = max(1, evals // pop)
        hits = []
        for run_seed in seeds:
            s = Settings(seed=seed, pop=pop, gens=gens, f=f, cr=cr, run_seed=run_seed)
            a = analyse(s, keep_history=False)
            hits.append(a.found_global)
        rows[label] = {"share_global": float(np.mean(hits)), "gens": gens, "pop": pop, "evals": pop * (gens + 1)}
    return {
        "de_small": rows["small"]["share_global"], "de_large": rows["large"]["share_global"],
        "de_small_evals": rows["small"]["evals"], "de_large_evals": rows["large"]["evals"],
        "ga_small": C.GA_SUCCESS_SMALL, "ga_large": C.GA_SUCCESS_LARGE,
        "cma_small": C.CMA_SUCCESS_SMALL, "cma_large": C.CMA_SUCCESS_LARGE,
    }


# --- Experiment 2: eigener Regler - Differenzgewicht F ----------------------------------------------------------------------------------------


def f_experiment(seed=None, values=None, seeds=None, gens=None, pop=None):
    seed = C.DEFAULT_SEED if seed is None else seed
    values = C.F_VALUES if values is None else values
    seeds = C.F_EXPERIMENT_SEEDS if seeds is None else seeds
    gens = C.F_EXPERIMENT_GENS if gens is None else gens
    pop = C.F_EXPERIMENT_POP if pop is None else pop

    rows = []
    for f in values:
        hits, final_divs = [], []
        for run_seed in seeds:
            s = Settings(seed=seed, pop=pop, gens=gens, f=f, run_seed=run_seed)
            r = run(s, keep_history=False)
            _, grid_xy, _ = instance(seed)
            hit = bool(np.sqrt(((r.best_individual - grid_xy) ** 2).sum()) <= C.GLOBAL_TOL_KM)
            hits.append(hit)
            final_divs.append(float(r.diversity_history[-1]))
        rows.append({"f": f, "share_global": float(np.mean(hits)), "diversity_end_median": float(np.median(final_divs))})
    return rows
