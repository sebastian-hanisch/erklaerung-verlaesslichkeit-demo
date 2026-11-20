"""Randfälle (DEMO-PLAYBOOK §12): M = 1, K = 1, konstante Spalte, Zeilenmenge mit einer Zeile, zwei identische Mengen, zwei Hintergrundzeilen, Reglergrenzen, Gleichstand an der Spitze."""

import warnings

import numpy as np
import pytest

import erv_constants as C
import erv_evaluation as ev
import erv_exact as E
import erv_rank as R
import erv_sampler as S
import erv_schichten as SC


def _tab(seed=0, nb=40):
    return np.random.default_rng(seed).normal(size=(64, nb))


def test_eine_einzige_reihenfolge_m_gleich_1():
    t = _tab()
    out = S.run_many(t, 1, 5, 1)
    assert out.shape == (5, 6) and np.isfinite(out).all()
    perms, z = S.draw(2, 1, 40)
    # eine Reihenfolge, eine Hintergrundzeile: der Zuwachs jedes Merkmals ist v(S + j, b) - v(S, b), die Summe v(voll, b) - v(leer, b)
    assert S.estimate(t, perms, z).sum() == pytest.approx(t[63, z[0]] - t[0, z[0]], abs=1e-12)
    assert S.cost(1) == 7


def test_ein_lauf_gibt_keine_paare_und_nicht_definiert_statt_null():
    s = ev.Settings(k=1)
    r = ev.erklaerer_lauf(s)
    assert r.laeufe.shape == (1, 6) and np.isnan(r.rang_laeufe)             # keine Paare von Läufen: nicht definiert, nicht 0
    assert R.pair_indices(1)[0].size == 0


def test_konstante_spalte_rangkorrelation_undefiniert_und_ausgelassen():
    phis = np.zeros((3, 4, 6))
    phis[:, :, 0] = 1.0                                                      # in jedem Modell und jeder Zeile sind alle Beiträge außer Art null: Beträge nicht konstant
    pm = R.paar_masse(phis)
    assert pm.shape == (3, 4)
    konst = np.ones((3, 4, 6))                                               # konstante Beträge: Spearman nicht definiert
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pm2 = R.paar_masse(konst)
    assert np.isnan(pm2[:, 0]).all() and np.isnan(pm2[:, 2]).all()
    assert (pm2[:, 1] == 1.0).all()                                          # Rang 1 bei Gleichstand: kleinster Index, also gleich
    assert np.isnan(SC.mittel(pm2)[0])                                       # alle undefiniert: Mittel nicht definiert (keine Warnung, kein 0)


def test_rang1_bei_gleichstand_ist_der_kleinste_index():
    assert R.rang1(np.array([0.5, 0.5, 0.1, 0.5, 0.0, 0.0])) == 0
    assert R.rang1(np.array([0.1, 0.5, 0.5])) == 1


def test_zeilenmenge_mit_einer_zeile_wichtigkeit_gleich_beitrag_betrag():
    phi = np.random.default_rng(1).normal(size=(1, 6))
    assert np.array_equal(R.wichtigkeit(phi), np.abs(phi[0]))


def test_zwei_identische_zeilenmengen_geben_rangkorrelation_1():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(20, 6))
    pm = R.paar_masse(np.stack([a, a]))
    assert pm[0, 0] == pytest.approx(1.0) and pm[0, 1] == 1.0 and pm[0, 2] == pytest.approx(1.0) and pm[0, 3] == 1.0


def test_zwei_hintergrundzeilen():
    f = lambda a: a[:, 0] * 2.0 + a[:, 1] * a[:, 2]   # noqa: E731
    rng = np.random.default_rng(3)
    X, B = rng.normal(size=(2, 6)), rng.normal(size=(2, 6))
    T = E.coalition_values(f, X, B)
    assert T.shape == (2, 64, 2)
    phi = E.exact_from_table(T)
    assert phi.sum(axis=1) == pytest.approx(T[:, 63, 0] - T[:, 0, :].mean(axis=1), abs=1e-12)
    out = S.run_many(T[0], 20, 3, 1)
    assert out.shape == (3, 6)


def test_gepaart_mit_ungerader_zahl_wird_abgelehnt():
    with pytest.raises(ValueError):
        S.run_many(_tab(), 11, 2, 1, paired=True)


def test_reglergrenzen_laufen_durch():
    for kw in (dict(m=C.M_MIN, k=C.K_MIN), dict(m=C.M_MAX, k=C.K_MAX), dict(neu=C.NEU_MIN, menge=C.MENGE_ALLE), dict(neu=C.NEU_MAX, menge=C.N_MENGEN - 1, ebene=C.EBENEN[1]),
               dict(zeile=C.ZEILE_MAX, erk_menge=C.N_TAB_MENGEN - 1, gepaart=True), dict(klasse="mlp", n=3000, seed=0), dict(klasse="rf", seed=C.SEED_MAX)):
        s = ev.Settings(**kw)
        nt = ev.neu_training(s)
        assert np.isfinite(nt.balken).all()
        r = ev.erklaerer_lauf(s)
        assert np.isfinite(r.l2) and 0 <= r.anders1 <= 1
        st = ev.stabil(s)
        assert len(st.art) == s.neu


def test_zwei_neu_trainings_ein_paar():
    s = ev.Settings(neu=2)
    nt = ev.neu_training(s)
    assert nt.n_paare == 1 and len(nt.paare) == 1


def test_leere_und_entartete_eingaben_der_masse():
    assert np.isnan(R.rel_l2(np.ones(6), np.zeros(6)))                       # Nenner 0: nicht definiert, keine Division
    assert np.isnan(R.rel_l2(np.zeros(6), np.zeros(6)))
    assert np.isnan(R.art_anteil(np.zeros(6)))                               # 0 / 0: nicht definiert, nicht 0 (nur in Mengen ohne Beitrag denkbar)
    assert R.art_anteil(np.array([1.0, 0, 0, 0, 0.0, 0])) == 1.0
