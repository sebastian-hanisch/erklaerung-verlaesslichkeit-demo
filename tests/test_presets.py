"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung, jedes Preset die erwartete Verdikt-Zeile (DEMO-PLAYBOOK 11a Punkt 7)."""

import pytest

import erv_constants as C
import erv_evaluation as ev
import erv_presets as P


def test_jedes_preset_hat_hilfetext_und_nur_bekannte_schluessel():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_werte_liegen_in_den_reglergrenzen():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_standard_preset_ist_die_standardeinstellung():
    assert C.PRESETS[next(iter(C.PRESETS))] == {}
    assert ev.Settings() == ev.Settings(**{k: v for k, v in C.PRESETS["Boosting, 300 Zeilen"].items()})


def test_setting_specs_decken_jeden_preset_schluessel_und_jeder_regler_hat_einen_schluessel():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert len(set(P.PRESET_KEYS.values())) == len(P.PRESET_KEYS)


def test_kein_preset_ist_zwei_mal_derselbe_zustand():
    zustaende = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(zustaende)) == len(zustaende)


def test_reglerschritte_geben_nur_gerade_m_und_zehnerschritte_fuer_laeufe():
    assert C.M_MIN % 2 == 0 and C.M_MAX % 2 == 0 and C.M_STEP % 2 == 0                # gepaarte Läufe brauchen gerades M
    assert C.K_MIN % C.K_STEP == 0 and C.M_MIN % C.M_STEP == 0 and (C.M_MAX - C.M_MIN) % C.M_STEP == 0


def test_alt_ist_ein_anderer_gueltiger_wert():
    for key, spec in P.SETTING_SPECS.items():
        assert spec.alt != spec.default, key
        if spec.options is not None:
            assert spec.alt in spec.options


def test_bool_caster_des_permalinks():
    assert P._bool("True") is True and P._bool("false") is False
    with pytest.raises(ValueError):
        P._bool("vielleicht")


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_hat_die_erwartete_verdikt_zeile(name):
    """Das Preset ändert nur, was es sagt: der Verdikt-Text nennt Klasse, Trainingsgröße und Reihenfolgen des Presets."""
    p = C.PRESETS[name]
    s = ev.Settings(klasse=p.get("klasse", C.DEFAULT_KLASSE), n=p.get("n", C.DEFAULT_N), menge=p.get("menge", C.DEFAULT_MENGE), m=p.get("m", C.DEFAULT_M))
    nt = ev.neu_training(s)
    kind, text = ev.verdict_neutraining(C.KLASSEN_KURZ[s.klasse], s.n, nt.beitrag_spanne, nt.wicht_spanne, nt.n_paare)
    assert C.KLASSEN_KURZ[s.klasse] in text and f"{s.n} Trainingszeilen" in text and "45 Paare je Menge" in text
    r = ev.erklaerer_lauf(s)
    _, t2 = ev.verdict_erklaerer(s.m, r.kosten, 2560, r.l2, r.anders1, s.k)
    assert f"**{s.m}** Reihenfolgen ({6 * s.m + 1} von 2560" in t2
    if name == "Wenige Reihenfolgen (30)":
        assert "(181 von 2560" in t2
