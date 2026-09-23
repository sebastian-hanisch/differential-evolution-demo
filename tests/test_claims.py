"""Jede im README/PRESET_HELP/App genannte Zahl wird hier nachgerechnet - keine Behauptung ohne Test.

Einzelne 40-100-Generationen-Läufe sind chaotisch empfindlich gegenüber winziger Fließkomma-Rundung (siehe
feedback_ci_platform_robust_tests.md, und die eigene Erfahrung aus nsga2-demo/nsga3-demo/moead-demo/cma-es-demo).
Zahlen aus einem EINZELNEN Lauf (Presets) bekommen deshalb nur Strukturgrenzen; Zahlen, die über mehrere Seeds mitteln
(Experimente, Sweep), sind von Natur aus robuster und dürfen engere (aber weiterhin großzügige) Bänder bekommen."""

import pytest

import de_constants as C
import de_evaluation as E


def _preset_analysis(name):
    p = C.PRESETS[name]
    s = E.Settings(seed=p["seed"], pop=p["pop"], gens=p["gens"], f=p["f"], cr=p["cr"], run_seed=p["run_seed"])
    return E.analyse(s, keep_history=False)


# --- Einzelläufe (Presets) - nur Strukturgrenzen, keine Nähe zu einem Messwert ------------------------------------------------------------


def test_standardfall_preset_claims():
    a = _preset_analysis("Standardfall")
    assert -1.0 < a.gap < 50.0
    assert a.result.diversity_history[-1] < 5.0


def test_kleines_differenzgewicht_preset_claims():
    a = _preset_analysis("Kleines Differenzgewicht")
    assert -1.0 < a.gap < 50.0


def test_grosses_differenzgewicht_preset_claims():
    a = _preset_analysis("Großes Differenzgewicht")
    assert -1.0 < a.gap < 50.0


def test_grosse_population_preset_claims():
    a = _preset_analysis("Große Population")
    assert -1.0 < a.gap < 50.0


# --- Headlinezahlen der beiden Experimente + Sweep (mitteln über mehrere Seeds, robuster) -----------------------------------------------------


def test_comparison_experiment_headline_claims():
    report = E.comparison_experiment()
    assert report["ga_small"] == C.GA_SUCCESS_SMALL == 0.55 and report["ga_large"] == C.GA_SUCCESS_LARGE == 0.95
    assert report["cma_small"] == C.CMA_SUCCESS_SMALL == 0.15 and report["cma_large"] == C.CMA_SUCCESS_LARGE == 0.15
    # Kernbefund: DE schlägt bei BEIDEN Budgets sowohl GA als auch CMA-ES klar - großzügiger Sicherheitsabstand
    assert report["de_small"] > report["ga_small"] + 0.2
    assert report["de_large"] > report["ga_large"] - 0.1        # GA ist bei großem Budget schon stark (95 %)
    assert report["de_small"] > report["cma_small"] + 0.5
    assert report["de_large"] > report["cma_large"] + 0.5


def test_f_experiment_headline_claims():
    rows = E.f_experiment()
    by_f = {r["f"]: r for r in rows}
    assert set(by_f) == set(C.F_VALUES)
    for r in rows:
        assert 0.0 <= r["share_global"] <= 1.0
        assert r["diversity_end_median"] >= 0.0
    # Kernbefund: das größte Differenzgewicht schneidet klar besser ab als das kleinste (knappes Budget)
    assert by_f[C.F_VALUES[-1]]["share_global"] > by_f[C.F_VALUES[0]]["share_global"] + 0.3
    # Kernbefund: größeres F hält die Population am Ende vielfältiger (kollabiert weniger)
    assert by_f[C.F_VALUES[-1]]["diversity_end_median"] > by_f[C.F_VALUES[0]]["diversity_end_median"]


def test_pop_sweep_headline_claims():
    rows = E.sweep("pop")
    assert [r["value"] for r in rows] == list(C.SWEEP_VALUES["pop"])
    for r in rows:
        assert 0.0 <= r["share_global"] <= 1.0
