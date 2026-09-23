"""Handrechnungen für Mutation/Crossover an einer konstruierten Population, End-to-End-Konvergenz auf einfachen
Testfunktionen (Kugel, Ellipsoid) - eigene Implementierung UND `scipy.optimize.differential_evolution`
(Referenzimplementierung, strategy='rand1bin' passend zur eigenen DE/rand/1/bin) im Vergleich."""

import numpy as np
import pytest
from scipy.optimize import differential_evolution

import de_algorithm as A


def test_mutate_matches_hand_calculation_with_fixed_indices(monkeypatch):
    pop = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])

    class FixedRng:
        def choice(self, candidates, size, replace):
            return np.array(sorted(candidates)[:3])   # deterministisch: immer die drei kleinsten Indizes != i

    mutants = A.mutate(pop, F=0.5, rng=FixedRng())
    # i=0: Kandidaten [1,2,3] -> r1,r2,r3 = 1,2,3 -> pop[1] + 0.5*(pop[2]-pop[3]) = [10,0] + 0.5*([0,10]-[10,10]) = [10,0]+0.5*[-10,0] = [5,0]
    assert mutants[0] == pytest.approx([5.0, 0.0])
    # i=1: Kandidaten [0,2,3] -> pop[0] + 0.5*(pop[2]-pop[3]) = [0,0] + 0.5*([0,10]-[10,10]) = [0,0]+0.5*[-10,0] = [-5,0]
    assert mutants[1] == pytest.approx([-5.0, 0.0])


def test_mutate_never_uses_the_target_index_itself():
    pop = np.random.default_rng(3).random((8, 2)) * 100
    rng = np.random.default_rng(9)
    for _ in range(20):
        mutants = A.mutate(pop, F=0.8, rng=rng)
        assert mutants.shape == pop.shape


def test_crossover_matches_hand_calculation_with_fixed_mask():
    pop = np.array([[1.0, 2.0, 3.0]])
    mutants = np.array([[10.0, 20.0, 30.0]])

    class FixedRng:
        def integers(self, lo, hi):
            return 0    # j_rand = Dimension 0 (ohnehin von mask erzwungen)

        def random(self, dim):
            return np.array([0.9, 0.1, 0.9])   # nur Dimension 1 kommt (ohne j_rand-Zwang) unter cr=0.5

    trial = A.crossover(pop, mutants, cr=0.5, rng=FixedRng())
    # mask = [0.9,0.1,0.9] < 0.5 = [False,True,False]; j_rand=0 erzwingt zusätzlich Dim 0 -> [True,True,False]
    assert trial[0] == pytest.approx([10.0, 20.0, 3.0])


def test_crossover_guarantees_at_least_one_dimension_from_the_mutant():
    pop = np.zeros((5, 4))
    mutants = np.ones((5, 4))
    rng = np.random.default_rng(1)
    trial = A.crossover(pop, mutants, cr=0.0, rng=rng)    # cr=0 -> ohne j_rand-Zwang bliebe alles Original
    assert np.all((trial == 1.0).sum(axis=1) >= 1)


def test_diversity_matches_hand_calculation():
    pop = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    # Schwerpunkt (5,5); jeder Punkt liegt sqrt(50) davon entfernt
    assert A.diversity(pop) == pytest.approx(np.sqrt(50.0))


def test_diversity_is_zero_for_a_collapsed_population():
    pop = np.full((6, 2), 3.0)
    assert A.diversity(pop) == pytest.approx(0.0)


# --- End-to-End: Konvergenz auf einfachen Testfunktionen (Kugel, elliptisch) --------------------------------------------------------------


def sphere(pop):
    return (np.asarray(pop) ** 2).sum(axis=-1)


def ellipsoid(pop, cond=100.0):
    pop = np.asarray(pop)
    dim = pop.shape[-1]
    scales = cond ** (np.arange(dim) / max(dim - 1, 1))
    return ((pop * scales) ** 2).sum(axis=-1)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_run_de_converges_on_sphere_function(seed):
    dim = 3
    r = A.run_de(sphere, dim, bounds=(-10.0, 10.0), pop_size=30, generations=200, F=0.8, cr=0.9, seed=seed)
    assert r.best_fitness < 1e-6
    assert np.all(np.diff(r.best_history) <= 1e-12)      # best_history ist monoton fallend (gierige Selektion)


def test_run_de_converges_on_ellipsoid_function():
    dim = 5
    r = A.run_de(ellipsoid, dim, bounds=(-5.0, 5.0), pop_size=50, generations=300, F=0.8, cr=0.9, seed=1)
    assert r.best_fitness < 1e-2


def test_run_de_and_scipy_reach_a_comparably_good_optimum_on_the_sphere():
    """Kein Generation-für-Generation-Gleichlauf (unterschiedliche RNG-Nutzung) - aber beide Implementierungen sollen
    auf derselben einfachen konvexen Funktion mit vergleichbarem Budget nahe an 0 landen."""
    dim = 3
    r = A.run_de(sphere, dim, bounds=(-10.0, 10.0), pop_size=30, generations=200, F=0.8, cr=0.9, seed=1)

    res = differential_evolution(
        lambda x: float(np.sum(np.asarray(x) ** 2)), bounds=[(-10.0, 10.0)] * dim,
        strategy="rand1bin", mutation=0.8, recombination=0.9, popsize=10, seed=1,
        init="random", maxiter=200, polish=False, tol=-1,
    )

    assert r.best_fitness < 1e-6
    assert res.fun < 1e-6


def test_run_de_respects_bounds():
    dim = 2
    r = A.run_de(sphere, dim, bounds=(-1.0, 1.0), pop_size=10, generations=50, F=1.5, cr=0.9, seed=1, keep_history=True)
    for g in r.generations:
        assert np.all(g.population >= -1.0) and np.all(g.population <= 1.0)


def test_run_de_history_shapes_and_generations_snapshots():
    dim = 2
    pop_size = 8
    gens = 12
    r = A.run_de(sphere, dim, bounds=(-10.0, 10.0), pop_size=pop_size, generations=gens, F=0.8, cr=0.9, seed=3, keep_history=True)
    assert r.best_history.shape == (gens + 1,)
    assert r.mean_history.shape == (gens + 1, dim)
    assert r.diversity_history.shape == (gens + 1,)
    assert len(r.generations) == gens + 1
    for g in r.generations:
        assert g.population.shape == (pop_size, dim)
        assert g.fitness.shape == (pop_size,)


def test_run_de_without_history_leaves_generations_empty():
    r = A.run_de(sphere, 2, bounds=(-10.0, 10.0), pop_size=8, generations=5, F=0.8, cr=0.9, seed=1, keep_history=False)
    assert r.generations == []
