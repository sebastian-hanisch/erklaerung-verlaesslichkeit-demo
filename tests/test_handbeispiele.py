"""Handbeispiele des Plans (Erklärer-Zufall, Abstand gegen Rang, Rangkorrelation, Zählungen, Zeilenmenge) als exakte Tests: Brüche und Ganzzahlen, zweiter Rechenweg getrennt.

Jede Erwartung steht als Rechnung oder als Bruch, den man von Hand nachrechnen kann; die Wahrscheinlichkeiten des Stichprobenschätzers kommen aus der Aufzählung aller Ausgänge
(Fraktionen), die Fremdreferenz ist scipy. Die Repo-Funktionen (`erv_sampler.estimate`, `erv_rank`, `erv_exact`) laufen auf dem Handspiel, eingebettet in sechs Merkmale (drei Nullspieler)."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest
from scipy.stats import rankdata, spearmanr

import erv_exact as E
import erv_rank as R
import erv_sampler as S

SPIEL = {frozenset(): 0, frozenset({0}): 3, frozenset({1}): 3, frozenset({2}): 2, frozenset({0, 1}): 4, frozenset({0, 2}): 4, frozenset({1, 2}): 3, frozenset({0, 1, 2}): 6}
PERMS3 = list(itertools.permutations(range(3)))


def zuwaechse(p):
    s, m = frozenset(), [0, 0, 0]
    for j in p:
        m[j] = SPIEL[s | {j}] - SPIEL[s]
        s = s | {j}
    return tuple(m)


def klasse(est, top=0):
    mx = max(est)
    w = [j for j in range(3) if est[j] == mx]
    if len(w) > 1:
        return "gleich"
    return "richtig" if w[0] == top else "falsch"


# ------------------------------------------------------------------------------------------------------------------ Teil 1: Erklärer-Zufall von Hand


def test_zuwaechse_je_reihenfolge_und_shapley_wert():
    m = [zuwaechse(p) for p in PERMS3]
    assert m == [(3, 1, 2), (3, 2, 1), (1, 3, 2), (3, 3, 0), (2, 2, 2), (3, 1, 2)]
    assert [F(sum(x[j] for x in m), 6) for j in range(3)] == [F(5, 2), F(2), F(3, 2)]


def test_repo_aufzaehlung_gibt_den_shapley_wert_des_handspiels(spiel3):
    phi = E.exact_from_table(spiel3[None])[0]
    assert phi.tolist() == pytest.approx([2.5, 2.0, 1.5, 0.0, 0.0, 0.0], abs=1e-12)


@pytest.mark.parametrize("m, erwartet", [
    (1, {"richtig": F(1, 2), "gleich": F(1, 3), "falsch": F(1, 6)}),
    (2, {"richtig": F(7, 12), "gleich": F(2, 9), "falsch": F(7, 36)}),
    (3, {"richtig": F(47, 72), "gleich": F(35, 216), "falsch": F(5, 27)}),
    (6, {"richtig": F(11933, 15552), "gleich": F(3839, 46656), "falsch": F(3509, 23328)}),
])
def test_verteilung_des_schaetzers_aus_allen_ausgaengen(m, erwartet):
    zw = [zuwaechse(p) for p in PERMS3]
    phi = [F(5, 2), F(2), F(3, 2)]
    z = {"richtig": 0, "gleich": 0, "falsch": 0}
    mean, sq = [F(0)] * 3, [F(0)] * 3
    for idx in itertools.product(range(6), repeat=m):
        est = [F(sum(zw[i][j] for i in idx), m) for j in range(3)]
        z[klasse(est)] += 1
        for j in range(3):
            mean[j] += est[j]
            sq[j] += est[j] ** 2
    tot = 6 ** m
    assert {k: F(v, tot) for k, v in z.items()} == erwartet
    mean = [x / tot for x in mean]
    assert mean == phi                                                   # erwartungstreu für jedes M
    for j in range(3):                                                  # Varianz = Varianz des Zuwachses / M (zweiter Rechenweg)
        var_zuwachs = sum((F(x[j]) - phi[j]) ** 2 for x in zw) / 6
        assert sq[j] / tot - mean[j] ** 2 == var_zuwachs / m


def test_bei_sechs_gezogenen_reihenfolgen_ist_rang1_nicht_sicher_aber_aufzaehlen_ist_exakt():
    assert F(11933, 15552) < 1 and float(F(11933, 15552)) == pytest.approx(0.767, abs=5e-4)


def test_gepaart_umkehrung_gibt_vier_von_sechs_richtig():
    zw = [zuwaechse(p) for p in PERMS3]
    z = {"richtig": 0, "gleich": 0, "falsch": 0}
    for i, p in enumerate(PERMS3):
        ir = PERMS3.index(tuple(reversed(p)))
        z[klasse([F(zw[i][j] + zw[ir][j], 2) for j in range(3)])] += 1
    assert z == {"richtig": 4, "gleich": 0, "falsch": 2}


@pytest.mark.parametrize("m, richtig, gleich, falsch", [(1, F(1, 2), F(1, 3), F(1, 6)), (2, F(7, 12), F(2, 9), F(7, 36))])
def test_repo_schaetzer_auf_dem_handspiel_gibt_dieselbe_verteilung(spiel3, m, richtig, gleich, falsch):
    """Der Schätzer des Repos (sechs Merkmale, drei Nullspieler) auf dem Handspiel: alle 720^M Ausgänge für M = 1 und 2 durchzählen. Die Reihenfolge der drei wirksamen Merkmale
    ist unter den 720 gleichverteilt, also dieselbe Verteilung wie bei 3! = 6 Reihenfolgen."""
    perms = np.array(list(itertools.permutations(range(6))))
    inc = np.array([S.estimate(spiel3, p[None, :], np.zeros(1, dtype=int))[:3] for p in perms])      # (720, 3) Zuwächse einer Reihenfolge
    assert inc.shape == (720, 3)
    if m == 1:
        summe = inc.astype(int)
    else:
        summe = (inc[:, None, :] + inc[None, :, :]).reshape(-1, 3).astype(int)    # Summe zweier Zuwächse (Mittel = Summe / 2: Vergleich über die Summen)
    mx = summe.max(axis=1, keepdims=True)
    n_top = (summe == mx).sum(axis=1)
    richtig_n = int(((n_top == 1) & (summe[:, 0] == mx[:, 0])).sum())
    gleich_n = int((n_top > 1).sum())
    tot = len(summe)
    assert F(richtig_n, tot) == richtig and F(gleich_n, tot) == gleich and F(tot - richtig_n - gleich_n, tot) == falsch


# ------------------------------------------------------------------------------------------------------------------ Teil 2: Abstand sieht nichts, Rang sieht es


def test_relativer_abstand_klein_aber_rang1_und_spearman_aendern_sich():
    phi = np.array([0.50, 0.49, 0.01])
    est = np.array([0.49, 0.50, 0.01])
    assert R.rel_l2(est, phi) == pytest.approx(0.0202, abs=5e-5)
    assert R.rel_l2(est, phi) == pytest.approx(math.sqrt(2 * 0.01 ** 2) / math.sqrt(0.25 + 0.2401 + 0.0001), rel=1e-12)
    assert R.rang1(phi) != R.rang1(est)
    assert R.spearman(phi, est) == pytest.approx(0.5, abs=1e-12)
    assert spearmanr(phi, est)[0] == pytest.approx(0.5, abs=1e-12)          # Fremdreferenz
    assert 1 - F(6 * 2, 3 * 8) == F(1, 2)                                   # 1 - 6 * Summe d^2 / (n (n^2 - 1)), Summe d^2 = 2
    assert np.corrcoef(phi, est)[0, 1] > 0.99                               # die Werte selbst korrelieren fast perfekt (Pearson, nicht Rang)


# ------------------------------------------------------------------------------------------------------------------ Teil 3: Rangkorrelation gegen scipy


def test_rangkorrelation_gegen_scipy_ohne_und_mit_gleichstaenden():
    rng = np.random.default_rng(1)
    for _ in range(50):
        a, b = rng.normal(size=6), rng.normal(size=6)
        assert R.spearman(a, b) == pytest.approx(spearmanr(a, b)[0], abs=1e-12)
        ra, rb = rankdata(a), rankdata(b)
        assert R.spearman(a, b) == pytest.approx(1 - 6 * np.sum((ra - rb) ** 2) / (6 * 35), abs=1e-12)      # Formel ohne Gleichstände
    n_geprueft = 0
    for _ in range(50):
        a, b = rng.integers(0, 3, 8).astype(float), rng.integers(0, 3, 8).astype(float)
        if np.std(a) == 0 or np.std(b) == 0:
            assert np.isnan(R.spearman(a, b))
            continue
        assert R.spearman(a, b) == pytest.approx(spearmanr(a, b)[0], abs=1e-12)
        n_geprueft += 1
    assert n_geprueft > 30


@pytest.mark.filterwarnings("ignore::scipy.stats.ConstantInputWarning")
def test_konstante_spalte_ist_undefiniert():
    assert np.isnan(R.spearman(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))
    assert np.isnan(spearmanr([1, 1, 1], [1, 2, 3])[0])


# ------------------------------------------------------------------------------------------------------------------ Teil 4: Zählungen


def test_zaehlungen_kosten_und_gleichstand():
    assert math.comb(10, 3) == sum(1 for _ in itertools.combinations(range(10), 3)) == 120
    assert sum(1 for _ in itertools.product((0, 1), repeat=6)) == 64 == 2 ** 6
    assert S.enumeration_cost() == E.K * 40 == 2560
    assert [S.cost(m) for m in (30, 100, 300)] == [181, 601, 1801]
    assert (S.enumeration_cost() - 1) // 6 == 426 and S.cost(426) < S.enumeration_cost() <= S.cost(427)    # ab 427 Reihenfolgen ist die Aufzählung billiger
    assert S.cost(426) == 2557 and S.cost(427) == 2563


# ------------------------------------------------------------------------------------------------------------------ Teil 5: Zeilenmenge von Hand

BEITRAG = {
    "A": {1: (4, 1, 2), 2: (1, 1, 4), 3: (3, 2, 4), 4: (1, 2, 5)},
    "B": {1: (5, 1, 2), 2: (3, 1, 1), 3: (2, 2, 4), 4: (0, 2, 6)},
}


def wichtigkeit3(m, zeilen):
    return [F(sum(BEITRAG[m][z][j] for z in zeilen), len(zeilen)) for j in range(3)]


def einbetten(w):
    """(Art, Kunde, Vorlauf) -> sechs Merkmale (Art, Kunde, Zoll 0, Wochenende 0, Vorlauf, Gewicht 0)."""
    return np.array([float(w[0]), float(w[1]), 0.0, 0.0, float(w[2]), 0.0])


def test_zeilenmenge_von_hand_wichtigkeit_rang1_rangkorrelation_anteil():
    S1, S2, VER = (1, 2), (3, 4), (1, 2, 3, 4)
    ia1, ib1 = wichtigkeit3("A", S1), wichtigkeit3("B", S1)
    ia2, ib2 = wichtigkeit3("A", S2), wichtigkeit3("B", S2)
    iaV, ibV = wichtigkeit3("A", VER), wichtigkeit3("B", VER)
    assert ia1 == [F(5, 2), F(1), F(3)] and ib1 == [F(4), F(1), F(3, 2)]
    assert ia2 == [F(2), F(2), F(9, 2)] and ib2 == [F(1), F(2), F(5)]
    assert iaV == [F(9, 4), F(3, 2), F(15, 4)] and ibV == [F(5, 2), F(3, 2), F(13, 4)]
    # Repo: Wichtigkeit als Mittel der Beträge über die Zeilen einer Menge
    phi_a1 = np.array([einbetten(BEITRAG["A"][z]) for z in S1])
    assert R.wichtigkeit(phi_a1)[[0, 1, 4]].tolist() == [2.5, 1.0, 3.0]
    # Rang 1: Menge 1 verschieden (Vorlauf gegen Art), Menge 2 und Vereinigung gleich (Vorlauf)
    assert (np.argmax([float(x) for x in ia1]), np.argmax([float(x) for x in ib1])) == (2, 0)
    assert (np.argmax([float(x) for x in ia2]), np.argmax([float(x) for x in ib2])) == (2, 2)
    assert (np.argmax([float(x) for x in iaV]), np.argmax([float(x) for x in ibV])) == (2, 2)
    # Rangkorrelation: 1/2 (Summe d^2 = 2), Wurzel 3 / 2 (mittlere Ränge bei Gleichstand), 1
    f = lambda w: [float(x) for x in w]   # noqa: E731
    assert R.spearman(np.array(f(ia1)), np.array(f(ib1))) == pytest.approx(0.5, abs=1e-12)
    assert R.spearman(np.array(f(ia2)), np.array(f(ib2))) == pytest.approx(math.sqrt(3) / 2, abs=1e-12)
    assert spearmanr(f(ia2), f(ib2))[0] == pytest.approx(math.sqrt(3) / 2, abs=1e-12)
    assert R.mean_ranks(np.array(f(ia2))).tolist() == [1.5, 1.5, 3.0] and R.mean_ranks(np.array(f(ib2))).tolist() == [1.0, 2.0, 3.0]
    assert R.spearman(np.array(f(iaV)), np.array(f(ibV))) == pytest.approx(1.0, abs=1e-12)
    # Anteil Art / (Art + Vorlauf)
    anteil = lambda w: w[0] / (w[0] + w[2])   # noqa: E731
    assert (anteil(ia1), anteil(ib1)) == (F(5, 11), F(8, 11))
    assert (anteil(ia2), anteil(ib2)) == (F(4, 13), F(1, 6))
    assert (anteil(iaV), anteil(ibV)) == (F(3, 8), F(10, 23))
    assert R.art_anteil(einbetten(ia1)) == pytest.approx(5 / 11, abs=1e-15) and R.art_anteil(einbetten(ib2)) == pytest.approx(1 / 6, abs=1e-15)
    # Wichtigkeit der Vereinigung = Mittel der Mengenwerte (gleich große Mengen)
    assert all(iaV[j] == (ia1[j] + ia2[j]) / 2 and ibV[j] == (ib1[j] + ib2[j]) / 2 for j in range(3))
    # Spannen über die Mengen: dieselben zwei Modelle, andere Menge, anderer Wert
    assert (min(anteil(ia1), anteil(ia2)), max(anteil(ia1), anteil(ia2))) == (F(4, 13), F(5, 11))
    assert (min(anteil(ib1), anteil(ib2)), max(anteil(ib1), anteil(ib2))) == (F(1, 6), F(8, 11))


def test_gleiches_rang1_je_zeile_in_drei_von_vier_zeilen():
    phi_a = np.array([einbetten(BEITRAG["A"][z]) for z in (1, 2, 3, 4)])
    phi_b = np.array([einbetten(BEITRAG["B"][z]) for z in (1, 2, 3, 4)])
    gleich = (R.rang1(phi_a) == R.rang1(phi_b)).tolist()
    assert gleich == [True, False, True, True]
    # Paar-Funktion des Repos: Anteil der Zeilen mit gleichem Rang 1 = 3/4
    pm = R.paar_masse(np.stack([phi_a, phi_b]))
    assert pm.shape == (1, 4) and pm[0, 1] == 0.75
