"""Orakel (anderer Rechenweg als der Code der Demo) für Aufzählung, Stichproben-Erklärer, Rangmaße und Zufallsstrom.

(a) Stichprobenschätzer mit ALLEN 720 Reihenfolgen x ALLEN Hintergrundzeilen = Koalitionsformel; (b) Koalitionsformel = Mittel über alle n! Reihenfolgen (exakte Brüche, n = 3 und 4),
(c) Rangkorrelation = scipy (mittlere Ränge), (d) Effizienz in jedem Lauf, (e) der NumPy-Zufallsstrom ist Bit für Bit der der Klasse SplitMix64, (f) Mittel vieler Läufe liegt im Band
um die Aufzählung (Standardfehler), (g) Linearfall in geschlossener Form."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest
from scipy.stats import spearmanr

import erv_exact as E
import erv_rank as R
import erv_sampler as S
from erv_rng import SplitMix64


def _zufallstabelle(seed, nb=40):
    return np.random.default_rng(seed).normal(size=(64, nb))


PERMS = np.array(list(itertools.permutations(range(6))))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_alle_reihenfolgen_mal_alle_hintergrundzeilen_gleich_koalitionsformel(seed):
    T = _zufallstabelle(seed)
    exakt = E.shapley_from_values(T.mean(axis=-1))
    P = np.repeat(PERMS, 40, axis=0)
    Z = np.tile(np.arange(40), 720)
    assert np.max(np.abs(S.estimate(T, P, Z) - exakt)) < 1e-12


def test_koalitionsformel_gleich_alle_reihenfolgen_exakte_brueche():
    """Zufallsspiele mit ganzzahligen Werten über n = 3 und n = 4 Spielern: Koalitionsformel gegen Mittel über alle n! Reihenfolgen, mit Brüchen (kein Rundungsrest)."""
    rng = np.random.default_rng(7)
    for n in (3, 4):
        for _ in range(20):
            v = {frozenset(c): int(rng.integers(-5, 6)) for r in range(n + 1) for c in itertools.combinations(range(n), r)}
            v[frozenset()] = 0
            koal = []
            for j in range(n):
                s = F(0)
                for r in range(n):
                    w = F(math.factorial(r) * math.factorial(n - r - 1), math.factorial(n))
                    for c in itertools.combinations([i for i in range(n) if i != j], r):
                        s += w * (v[frozenset(c) | {j}] - v[frozenset(c)])
                koal.append(s)
            perm = [F(0)] * n
            for p in itertools.permutations(range(n)):
                cur = frozenset()
                for j in p:
                    perm[j] += v[cur | {j}] - v[cur]
                    cur = cur | {j}
            assert koal == [x / math.factorial(n) for x in perm]


def test_repo_aufzaehlung_gleich_brueche_auf_sechs_spielern():
    """Die Aufzählung des Repos (float) gegen die Brüche der Koalitionsformel für sechs Spieler mit ganzzahligen Werten."""
    rng = np.random.default_rng(11)
    for _ in range(5):
        vals = rng.integers(-6, 7, size=64).astype(float)
        vals[0] = 0.0
        ref = []
        for j in range(6):
            s = F(0)
            for k in range(64):
                if (k >> j) & 1:
                    continue
                r = bin(k).count("1")
                s += F(math.factorial(r) * math.factorial(5 - r), math.factorial(6)) * (int(vals[k | (1 << j)]) - int(vals[k]))
            ref.append(float(s))
        assert E.shapley_from_values(vals).tolist() == pytest.approx(ref, abs=1e-12)


def test_effizienz_der_aufzaehlung_und_jedes_laufs():
    T = _zufallstabelle(3)
    assert E.efficiency_gap(T[None]) < 1e-12
    for seed in range(5):
        perms, z = S.draw(seed, 50, 40)
        est = S.estimate(T, perms, z)
        # Summe der Beiträge = Mittel über die gezogenen Hintergrundzeilen von v_b(voll) - v_b(leer) (bei einer echten Wertetabelle: f(x) minus Mittel von f über die gezogenen Zeilen)
        assert est.sum() == pytest.approx((T[63, z] - T[0, z]).mean(), abs=1e-12)


def test_zufallsstrom_numpy_gleich_klasse_bit_fuer_bit():
    for seed in (0, 12345, 2 ** 63 + 5):
        r = SplitMix64(seed)
        ref = np.array([r.next_u64() for _ in range(300)], dtype=np.uint64)
        assert (S.u64_block(seed, 300) == ref).all()
        assert (S.u64_block(seed, 50, start=100) == ref[100:150]).all()
    p1, z1 = S.draw(777, 120, 40)
    p2, z2 = S.draw_reference(777, 120, 40)
    assert (p1 == p2).all() and (z1 == z2).all()
    assert all(sorted(p) == list(range(6)) for p in p1)


def test_mittel_vieler_laeufe_liegt_im_band_um_die_aufzaehlung():
    T = _zufallstabelle(5)
    exakt = E.shapley_from_values(T.mean(axis=-1))
    M = 100
    n_laeufe = 3000
    laeufe = S.run_many(T, M, n_laeufe, 99)
    se = laeufe.std(axis=0, ddof=1) / math.sqrt(n_laeufe)
    z = (laeufe.mean(axis=0) - exakt) / se
    assert np.max(np.abs(z)) < 4.5            # erwartungstreu; das Band ist 4,5 Standardfehler (sechs Merkmale)


def test_streuung_faellt_mit_m_im_mittel_ueber_zeilen():
    rng = np.random.default_rng(3)
    tabs = rng.normal(size=(8, 64, 40))
    sds = []
    for m in (10, 40, 160):
        s = [S.run_many(t, m, 60, 5).std(axis=0, ddof=1).mean() for t in tabs]
        sds.append(float(np.mean(s)))
    assert sds[0] > sds[1] > sds[2]


def test_linearer_fall_geschlossene_form():
    """f(x) = w * x: der interventionale Beitrag ist w_j (x_j - Mittel der Hintergrundspalte j) (Linear-SHAP), unabhängig von der Reihenfolge."""
    rng = np.random.default_rng(2)
    w = rng.normal(size=6)
    X = rng.normal(size=(3, 6))
    B = rng.normal(size=(40, 6))
    T = E.coalition_values(lambda a: a @ w, X, B)
    assert E.exact_from_table(T) == pytest.approx(w * (X - B.mean(axis=0)), abs=1e-12)
    # jeder einzelne Stichprobenlauf mit einer Reihenfolge liegt bei w_j (x_j - b_j) der gezogenen Hintergrundzeile
    perms, z = S.draw(3, 1, 40)
    est = S.estimate(T[0], perms, z)
    assert est == pytest.approx(w * (X[0] - B[z[0]]), abs=1e-12)


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_rangkorrelation_gegen_scipy_100_zufallsvektoren_mit_und_ohne_gleichstaende():
    rng = np.random.default_rng(4)
    n_def = 0
    for i in range(100):
        a = rng.normal(size=6) if i < 50 else rng.integers(0, 4, 6).astype(float)
        b = rng.normal(size=6) if i < 50 else rng.integers(0, 4, 6).astype(float)
        ref = spearmanr(a, b)[0]
        got = R.spearman(a, b)
        if np.isnan(ref):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(ref, abs=1e-12)
            n_def += 1
    assert n_def > 80


def test_rangkorrelation_stapelweise_gleich_einzeln():
    rng = np.random.default_rng(8)
    A, B = rng.normal(size=(7, 5, 6)), rng.normal(size=(7, 5, 6))
    got = R.spearman(A, B)
    for i in range(7):
        for j in range(5):
            assert got[i, j] == pytest.approx(R.spearman(A[i, j], B[i, j]), abs=1e-15)
