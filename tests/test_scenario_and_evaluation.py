"""Vehikel (Reproduzierbarkeit, bitidentisch zu genetic-algorithm-demo/cma-es-demo) und Auswertung (Kosten, Sweep,
beide Experimente) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import de_constants as C
import de_evaluation as E
import de_scenario as S


def test_generate_real_is_reproducible_and_shaped():
    a = S.generate_real(seed=7)
    b = S.generate_real(seed=7)
    assert np.array_equal(a.centres, b.centres) and np.array_equal(a.amplitudes, b.amplitudes) and np.array_equal(a.sigmas, b.sigmas)
    assert a.centres.shape == (C.K_WELLS, 2)


def test_generate_real_matches_genetic_algorithm_demo_bit_for_bit():
    """Bei Standard-Vehikel-Seed 35 muss die Landschaft bitidentisch zu genetic-algorithm-demo/cma-es-demo sein -
    reproduziert deren generate_real hier lokal (kein Cross-Repo-Import, wie überall im Portfolio)."""
    def ref_generate_real(seed):
        rng = np.random.default_rng(seed)
        centres = C.WELL_MARGIN + rng.random((C.K_WELLS, 2)) * (C.AREA - 2 * C.WELL_MARGIN)
        amplitudes = rng.uniform(C.AMP_MIN, C.AMP_MAX, size=C.K_WELLS)
        sigmas = rng.uniform(C.SIGMA_MIN, C.SIGMA_MAX, size=C.K_WELLS)
        return centres, amplitudes, sigmas

    centres_ref, amp_ref, sigma_ref = ref_generate_real(C.DEFAULT_SEED)
    inst = S.generate_real(C.DEFAULT_SEED)
    assert np.array_equal(inst.centres, centres_ref)
    assert np.array_equal(inst.amplitudes, amp_ref)
    assert np.array_equal(inst.sigmas, sigma_ref)


def test_cost_matches_manual_gaussian_well_computation():
    inst = S.generate_real(seed=3)
    xy = np.array([[50.0, 50.0], [10.0, 10.0]])
    d2 = ((xy[:, None, :] - inst.centres[None, :, :]) ** 2).sum(axis=-1)
    expected = inst.offset - (inst.amplitudes[None, :] * np.exp(-d2 / (2.0 * inst.sigmas[None, :] ** 2))).sum(axis=-1)
    assert inst.cost(xy) == pytest.approx(expected)


def test_grid_optimum_is_near_the_deepest_well_and_beats_all_well_centres():
    """Das Gitter trifft die exakte Muldenmitte nie genau (Diskretisierungsfehler in Höhe von GRID_STEP)."""
    inst = S.generate_real(seed=C.DEFAULT_SEED)
    grid_xy, grid_cost = S.grid_optimum(inst)
    centre_costs = inst.cost(inst.centres)
    assert grid_cost <= centre_costs.min() + 1.0
    assert grid_cost < inst.offset


# --- Auswertung --------------------------------------------------------------------------------------------------------------------------------


def test_cost_fn_matches_instance_cost_directly():
    s = E.Settings(seed=5)
    cost, grid_xy, grid_cost = E.cost_fn(s)
    pop = np.array([[10.0, 20.0], [80.0, 90.0]])
    inst, _, _ = E.instance(5)
    assert cost(pop) == pytest.approx(inst.cost(pop))


def test_run_returns_expected_shapes():
    s = E.Settings(gens=10, pop=8)
    r = E.run(s, keep_history=True)
    assert r.best_history.shape == (11,)
    assert len(r.generations) == 11


def test_run_respects_bounds():
    s = E.Settings(gens=20, pop=8, f=1.8)
    r = E.run(s, keep_history=True)
    for g in r.generations:
        assert np.all(g.population >= C.BOUNDS[0]) and np.all(g.population <= C.BOUNDS[1])


def test_analyse_gap_and_found_global_are_consistent():
    a = E.analyse(E.Settings(gens=30))
    assert np.isfinite(a.gap)
    dist = np.sqrt(((a.result.best_individual - a.grid_xy) ** 2).sum())
    assert a.found_global == (dist <= C.GLOBAL_TOL_KM)


def test_sweep_smoke():
    rows = E.sweep("pop", base=E.Settings(gens=20), values=(6, 10))
    assert len(rows) == 2
    assert all(0.0 <= r["share_global"] <= 1.0 for r in rows)


def test_comparison_experiment_smoke_small():
    report = E.comparison_experiment(seeds=(1, 2))
    assert 0.0 <= report["de_small"] <= 1.0 and 0.0 <= report["de_large"] <= 1.0
    assert report["ga_small"] == C.GA_SUCCESS_SMALL and report["ga_large"] == C.GA_SUCCESS_LARGE
    assert report["cma_small"] == C.CMA_SUCCESS_SMALL and report["cma_large"] == C.CMA_SUCCESS_LARGE
    assert report["de_small_evals"] > 0 and report["de_large_evals"] > report["de_small_evals"]


def test_f_experiment_smoke_small():
    rows = E.f_experiment(values=(0.1, 1.5), seeds=(1, 2), gens=15, pop=8)
    assert len(rows) == 2
    assert all(0.0 <= r["share_global"] <= 1.0 for r in rows)
    assert all(r["diversity_end_median"] >= 0.0 for r in rows)
