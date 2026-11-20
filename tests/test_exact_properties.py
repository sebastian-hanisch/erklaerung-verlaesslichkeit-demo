"""Eigenschaften der Aufzählung und des Stichproben-Erklärers (je 20 bis 50 Zufallsinstanzen, fester Zufalls-Seed)."""

import numpy as np
import pytest

import erv_exact as E
import erv_sampler as S


def _modell(seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=6)
    q = rng.normal(size=(6, 6))
    return lambda a: a @ w + np.sin((a @ q).sum(axis=1)) + (a[:, 0] * a[:, 1] * a[:, 4])


def test_umbenennen_und_umsortieren_der_merkmale_aendert_nur_die_reihenfolge():
    rng = np.random.default_rng(1)
    for s in range(20):
        f = _modell(s)
        X, B = rng.normal(size=(2, 6)), rng.normal(size=(12, 6))
        phi = E.exact_from_table(E.coalition_values(f, X, B))
        perm = rng.permutation(6)
        f2 = lambda a, perm=perm: f(a[:, np.argsort(perm)])   # noqa: E731  Modell auf umsortierten Spalten
        phi2 = E.exact_from_table(E.coalition_values(f2, X[:, perm], B[:, perm]))
        assert phi2 == pytest.approx(phi[:, perm], abs=1e-10)


def test_verschiebung_der_zielgroesse_aendert_keinen_beitrag():
    rng = np.random.default_rng(2)
    for s in range(20):
        f = _modell(s)
        X, B = rng.normal(size=(2, 6)), rng.normal(size=(12, 6))
        phi = E.exact_from_table(E.coalition_values(f, X, B))
        phi2 = E.exact_from_table(E.coalition_values(lambda a: f(a) + 17.5, X, B))
        assert phi2 == pytest.approx(phi, abs=1e-10)


def test_nicht_gelesenes_merkmal_hat_beitrag_exakt_null():
    rng = np.random.default_rng(3)
    for s in range(20):
        w = rng.normal(size=6)
        w[5] = 0.0
        f = lambda a, w=w: np.tanh(a[:, :5] @ w[:5]) + a[:, 0] * a[:, 3]   # noqa: E731  liest Spalte 5 (Gewicht) nie
        X, B = rng.normal(size=(2, 6)), rng.normal(size=(12, 6))
        phi = E.exact_from_table(E.coalition_values(f, X, B))
        assert np.all(phi[:, 5] == 0.0)
        # derselbe Nullbeitrag auch im Schätzer: jede Stichprobe liefert für das Merkmal den Zuwachs 0
        t = E.coalition_values(f, X[:1], B)[0]
        assert np.all(S.run_many(t, 20, 10, 4)[:, 5] == 0.0)


def test_gleicher_zufalls_seed_gleiche_laeufe_anderer_seed_andere():
    t = np.random.default_rng(5).normal(size=(64, 40))
    a, b, c = S.run_many(t, 50, 8, 3), S.run_many(t, 50, 8, 3), S.run_many(t, 50, 8, 4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_gepaarte_laeufe_nutzen_umkehrung_und_gleiche_hintergrundzeile():
    perms, z = S.draw(9, 5, 40)
    t = np.random.default_rng(6).normal(size=(64, 40))
    ref = S.estimate(t, np.concatenate([perms, perms[:, ::-1]]), np.concatenate([z, z]))
    # run_many mit Seed-Mischung: dasselbe Paar-Verfahren, nur anderer Strom; hier der Aufbau direkt
    out = S.run_many(t, 10, 1, 9, paired=True)
    assert out.shape == (1, 6)
    perms2, z2 = S.draw(S.mix_seed(9, 0), 5, 40)
    ref2 = S.estimate(t, np.concatenate([perms2, perms2[:, ::-1]]), np.concatenate([z2, z2]))
    assert out[0] == pytest.approx(ref2, abs=1e-14) and ref.shape == (6,)


def test_vereinigung_gleich_grosser_zeilenmengen_hat_als_wichtigkeit_das_mittel_der_mengenwerte():
    import erv_rank as R
    rng = np.random.default_rng(7)
    for _ in range(20):
        a, b = rng.normal(size=(20, 6)), rng.normal(size=(20, 6))
        assert R.wichtigkeit(np.vstack([a, b])) == pytest.approx((R.wichtigkeit(a) + R.wichtigkeit(b)) / 2, abs=1e-14)
