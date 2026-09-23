"""Presets: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session
(Permalink-Klammern und Preset-Knöpfe werden über AppTest in test_app.py geprüft, wie im Rest des Portfolios üblich)."""

import de_constants as C
import de_evaluation as E
import de_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.POP_MIN <= p["pop"] <= C.POP_MAX
        assert C.GEN_MIN <= p["gens"] <= C.GEN_MAX
        assert C.F_MIN <= p["f"] <= C.F_MAX
        assert C.CR_MIN <= p["cr"] <= C.CR_MAX
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            spec.caster(p[key])


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Standardfall"]
    s = E.Settings(seed=p["seed"], pop=p["pop"], gens=p["gens"], f=p["f"], cr=p["cr"], run_seed=p["run_seed"])
    assert s == E.Settings()


def test_bounds_and_steps_constants():
    assert P.bounds("pop_slider") == (C.POP_MIN, C.POP_MAX)
    assert P.bounds("f_slider") == (C.F_MIN, C.F_MAX)
    assert P.bounds("seed_input") == (0, C.SEED_MAX)
    assert set(P.STEPS) == {"pop_slider", "gens_slider", "f_slider", "cr_slider"}


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_large_population_preset_uses_a_bigger_pop_than_default():
    assert C.PRESETS["Große Population"]["pop"] > C.DEFAULT_POP
