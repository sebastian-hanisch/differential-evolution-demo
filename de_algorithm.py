"""DE-Kern: klassisches DE/rand/1/bin (Storn & Price, 1997). Kein GA-/CMA-ES-Kern kopiert - andere Mechanik: die
Mutation nutzt die aktuelle Streuung der Population SELBST als Schrittweiten-Signal (skalierter Differenzvektor
zweier zufälliger Populationsmitglieder), keine explizit gelernte Kovarianzmatrix wie bei CMA-ES."""

from dataclasses import dataclass, field

import numpy as np


def mutate(pop, F, rng):
    """Für jedes Individuum i: drei von i und voneinander verschiedene Indizes r1,r2,r3; Mutant = x_r1 + F*(x_r2-x_r3).
    Braucht mindestens 4 Individuen in der Population."""
    n = len(pop)
    mutants = np.empty_like(pop)
    for i in range(n):
        candidates = np.array([j for j in range(n) if j != i])
        r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
        mutants[i] = pop[r1] + F * (pop[r2] - pop[r3])
    return mutants


def crossover(pop, mutants, cr, rng):
    """Binomiale Rekombination: jede Dimension des Kindes stammt mit Wahrscheinlichkeit cr vom Mutanten, sonst vom
    Original - mindestens eine Dimension (j_rand) stammt garantiert vom Mutanten, damit das Kind nie exakt dem
    Original entspricht."""
    n, dim = pop.shape
    trial = pop.copy()
    for i in range(n):
        j_rand = rng.integers(0, dim)
        mask = rng.random(dim) < cr
        mask[j_rand] = True
        trial[i, mask] = mutants[i, mask]
    return trial


def diversity(pop):
    """Mittlerer euklidischer Abstand vom Populationsschwerpunkt (wie genetic-algorithm-demos diversity_real)."""
    centre = pop.mean(axis=0)
    return float(np.mean(np.sqrt(((pop - centre) ** 2).sum(axis=1))))


@dataclass
class Generation:
    population: np.ndarray
    fitness: np.ndarray


@dataclass
class DEResult:
    best_individual: np.ndarray
    best_fitness: float
    best_history: np.ndarray       # (generations + 1,) - bester bisher gefundener Wert je Generation (0 = Startpopulation)
    mean_history: np.ndarray       # (generations + 1, dim)
    diversity_history: np.ndarray  # (generations + 1,)
    generations: list = field(default_factory=list)   # nur befüllt, wenn keep_history=True


def run_de(cost_fn, dim, bounds, pop_size, generations, F, cr, seed, keep_history=False):
    """Ein DE-Lauf. `cost_fn(population) -> (pop_size,)`, niedriger ist besser. `bounds`: (lo, hi), gilt für jede
    Dimension - die Population wird nach jeder Mutation/jedem Crossover auf diese Box zurückgeschnitten."""
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    pop = lo + rng.random((pop_size, dim)) * (hi - lo)
    fitness = np.asarray(cost_fn(pop))

    best_idx = int(np.argmin(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    best_history = [best_f]
    mean_history = [pop.mean(axis=0)]
    diversity_history = [diversity(pop)]
    gens_snapshots = [Generation(pop.copy(), fitness.copy())] if keep_history else []

    for _ in range(generations):
        mutants = np.clip(mutate(pop, F, rng), lo, hi)
        trial = np.clip(crossover(pop, mutants, cr, rng), lo, hi)
        trial_fitness = np.asarray(cost_fn(trial))

        improved = trial_fitness <= fitness
        pop = np.where(improved[:, None], trial, pop)
        fitness = np.where(improved, trial_fitness, fitness)

        gen_best_idx = int(np.argmin(fitness))
        if fitness[gen_best_idx] < best_f:
            best_f = float(fitness[gen_best_idx])
            best_x = pop[gen_best_idx].copy()

        best_history.append(best_f)
        mean_history.append(pop.mean(axis=0))
        diversity_history.append(diversity(pop))
        if keep_history:
            gens_snapshots.append(Generation(pop.copy(), fitness.copy()))

    return DEResult(best_x, best_f, np.array(best_history), np.array(mean_history), np.array(diversity_history), gens_snapshots)
