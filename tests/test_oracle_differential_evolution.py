"""Orakel-Tests (unabhängiger Rechenweg): Mutation über eine Dekodierung per Basis-7-Population (F=2, welches geordnete
Tripel r1,r2,r3 wurde gezogen?), Crossover-Häufigkeiten gegen cr + (1-cr)/dim, Konsistenz jedes Generationsschritts
(Zulässigkeit der Grenzen, gierige Selektion, Kind ist ein gültiger Trial aus Tripel + Maske) und eine
Nachsimulation des gesamten Laufs mit Schleifen und demselben Zufallsstrom."""

import itertools
import math

import numpy as np
import pytest

import de_algorithm as A


def sphere(P):
    return (np.asarray(P) ** 2).sum(axis=-1)


def rastrigin(P):
    P = np.asarray(P)
    return 10 * P.shape[-1] + (P ** 2 - 10 * np.cos(2 * np.pi * P)).sum(axis=-1)


def test_mutation_draws_valid_distinct_triples_uniformly():
    n, F, N = 6, 2.0, 2500
    pop = np.array([[7.0 ** j] for j in range(n)])
    dec = {round(pop[a][0] + F * (pop[b][0] - pop[c][0]), 6): (a, b, c) for a, b, c in itertools.permutations(range(n), 3)}
    assert len(dec) == n * (n - 1) * (n - 2)               # Dekodierung eindeutig
    rng = np.random.default_rng(1)
    cnt = [dict() for _ in range(n)]
    for _ in range(N):
        mut = A.mutate(pop, F, rng)
        for i in range(n):
            t = dec[round(mut[i][0], 6)]
            assert i not in t and len(set(t)) == 3
            cnt[i][t] = cnt[i].get(t, 0) + 1
    for i in range(n):
        trip = [t for t in itertools.permutations(range(n), 3) if i not in t]
        p = 1 / len(trip)
        for t in trip:
            assert abs(cnt[i].get(t, 0) / N - p) < 5.5 * math.sqrt(p * (1 - p) / N)


@pytest.mark.parametrize("dim,cr", [(1, 0.0), (3, 0.0), (3, 0.3), (6, 0.9), (4, 1.0)])
def test_crossover_frequencies_match_cr_plus_jrand(dim, cr):
    rng = np.random.default_rng(2)
    pop, mut, N = np.zeros((1, dim)), np.arange(1, dim + 1, dtype=float)[None, :], 6000
    from_mut = np.zeros(dim)
    for _ in range(N):
        m = A.crossover(pop, mut, cr, rng)[0] != 0
        assert m.any()                                       # j_rand erzwingt mindestens eine Dimension
        from_mut += m
    p = cr + (1 - cr) / dim
    assert from_mut / N == pytest.approx(p, abs=5.5 * math.sqrt(max(p * (1 - p), 1e-9) / N) + 1e-9)


def _valid_trial_exists(old, new, i, F, lo, hi):
    for r1, r2, r3 in itertools.permutations([j for j in range(len(old)) if j != i], 3):
        mut = np.clip(old[r1] + F * (old[r2] - old[r3]), lo, hi)
        from_mut = np.isclose(new[i], mut, atol=1e-12, rtol=0)
        same = np.isclose(new[i], old[i], atol=1e-12, rtol=0)
        if np.all(from_mut | same) and from_mut.any():
            return True
    return False


def test_every_generation_step_is_a_valid_greedy_selection():
    rng = np.random.default_rng(3)
    for t in range(40):
        dim, npop, gens = int(rng.integers(1, 4)), int(rng.integers(4, 8)), int(rng.integers(2, 8))
        F, cr = float(rng.choice([0.0, 0.8, 1.5, 2.0])), float(rng.choice([0.0, 0.5, 1.0]))
        lo, hi = float(rng.choice([-5.0, 0.0])), float(rng.choice([5.0, 100.0]))
        f = sphere if t % 2 else rastrigin
        res = A.run_de(f, dim, (lo, hi), npop, gens, F, cr, int(rng.integers(0, 10 ** 6)), keep_history=True)
        best = float("inf")
        for k, g in enumerate(res.generations):
            assert g.population.min() >= lo and g.population.max() <= hi
            assert g.fitness == pytest.approx(f(g.population))
            best = min(best, g.fitness.min())
            assert res.best_history[k] == pytest.approx(best, abs=1e-12)
            centre = g.population.mean(axis=0)
            div = sum(math.sqrt(sum((x - c) ** 2 for x, c in zip(row, centre))) for row in g.population) / npop
            assert res.diversity_history[k] == pytest.approx(div, abs=1e-9)
            if k:
                old = res.generations[k - 1]
                assert np.all(g.fitness <= old.fitness + 1e-15)
                for i in range(npop):
                    if not np.array_equal(g.population[i], old.population[i]):
                        assert _valid_trial_exists(old.population, g.population, i, F, lo, hi)


def test_full_run_matches_loop_reimplementation_with_same_random_stream():
    rng0 = np.random.default_rng(4)
    for t in range(15):
        dim, npop, gens = int(rng0.integers(1, 4)), int(rng0.integers(4, 10)), int(rng0.integers(2, 8))
        F, cr = float(rng0.choice([0.1, 0.8, 1.5])), float(rng0.choice([0.0, 0.5, 0.9]))
        f = sphere if t % 2 else rastrigin
        seed = int(rng0.integers(0, 10 ** 6))
        res = A.run_de(f, dim, (-5.0, 5.0), npop, gens, F, cr, seed, keep_history=True)
        r = np.random.default_rng(seed)
        pop = -5.0 + r.random((npop, dim)) * 10.0
        fit = f(pop)
        for _ in range(gens):
            mut = np.empty_like(pop)
            for i in range(npop):
                cand = [j for j in range(npop) if j != i]
                a, b, c = (cand[p] for p in r.choice(len(cand), size=3, replace=False))
                mut[i] = pop[a] + F * (pop[b] - pop[c])
            mut = np.clip(mut, -5.0, 5.0)
            trial = pop.copy()
            for i in range(npop):
                jr = r.integers(0, dim)
                u = r.random(dim)
                for d in range(dim):
                    if u[d] < cr or d == jr:
                        trial[i, d] = mut[i, d]
            trial = np.clip(trial, -5.0, 5.0)
            ft = f(trial)
            for i in range(npop):
                if ft[i] <= fit[i]:
                    pop[i], fit[i] = trial[i], ft[i]
        assert res.generations[-1].population == pytest.approx(pop)
        assert res.generations[-1].fitness == pytest.approx(fit)
