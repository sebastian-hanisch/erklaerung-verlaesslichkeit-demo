"""Wahrheit je Funktional: Beitrag der wahren Funktion mu unter dem interventionalen Funktional, mit der exakten Bevölkerung als Hintergrund.

„Exakte Bevölkerung“ heißt hier die analytische Rechnung aus dem Generator: die gemeinsame Verteilung der fünf wirksamen Merkmale (Art, Kunde, Zoll, Wochenende, Vorlauf) als gewichtete
Stützpunkte, der Vorlauf mit Gauß-Legendre-Knoten (Export gleichverteilt in [1, 10], Umschlag in [0,5; 6]); das Gewicht ist ein reines Rauschmerkmal, ein Nullspieler, und fällt weg.
Die „Referenzmenge“ ist dagegen eine große Stichprobe (4 000 Instanzen) als Menge der erklärten Zeilen.
Art-Anteil = Anteil der Art an (Art + Vorlauf) der Wichtigkeit (Mittel von |Beitrag|)."""

from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss

import erv_generator as G

D5 = 5
_MASKS = np.array([[(k >> j) & 1 for j in range(D5)] for k in range(1 << D5)], dtype=bool)
_SIZE = _MASKS.sum(axis=1)
_W = np.array([factorial(s) * factorial(D5 - s - 1) / factorial(D5) if s < D5 else 0.0 for s in range(D5 + 1)])


def _vorlauf_knoten(lo, hi, q):
    x, w = leggauss(q)
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo), 0.5 * w


def bevoelkerung(q=24):
    """(X (N, 6), w (N,)): gewichtete Stützpunkte der Verteilung der Merkmale; Gewichte summieren zu 1, die Gewicht-Spalte ist 0."""
    X, w = [], []
    for a in range(4):
        for k in range(G.K_KUNDEN):
            zs = [(0, 1.0)] if a != 0 else [(0, 1 - G.P_ZOLL), (1, G.P_ZOLL)]
            for z, pz in zs:
                for we, pw in ((0, 5 / 7), (1, 2 / 7)):
                    if a == 1:
                        vs, wv = _vorlauf_knoten(1.0, 10.0, q)
                    elif a == 2:
                        vs, wv = _vorlauf_knoten(0.5, 6.0, q)
                    else:
                        vs, wv = np.array([0.0]), np.array([1.0])
                    for v, wvv in zip(vs, wv):
                        X.append([a, k, z, we, v, 0.0])
                        w.append(G.ART_P[a] / G.K_KUNDEN * pz * pw * wvv)
    X = np.array(X, float)
    w = np.array(w, float)
    assert abs(w.sum() - 1) < 1e-12
    return X, w


def shapley5(v):
    """v (n, 32) Koalitionswerte -> Beiträge (n, 5) nach der Koalitionsformel (fünf Spieler)."""
    phi = np.zeros((v.shape[0], D5))
    for j in range(D5):
        for k in range(1 << D5):
            if (k >> j) & 1:
                continue
            phi[:, j] += _W[_SIZE[k]] * (v[:, k | (1 << j)] - v[:, k])
    return phi


def werte(f, X, Xb, wb, chunk=400_000):
    """Interventionale Koalitionswerte (n, 32): v(S) = Summe_b w_b f(x_S, b_rest) für f auf den sechs Spalten (Gewicht = 0)."""
    n, nb = len(X), len(Xb)
    out = np.zeros((n, 1 << D5))
    step = max(1, chunk // ((1 << D5) * nb))
    for a in range(0, n, step):
        xs = X[a:a + step]
        H = np.where(_MASKS[None, :, None, :], xs[:, None, None, :D5], Xb[None, None, :, :D5])
        Z = np.zeros(H.shape[:-1] + (6,))
        Z[..., :D5] = H
        out[a:a + step] = f(Z.reshape(-1, 6)).reshape(len(xs), 1 << D5, nb) @ wb
    return out


def exakte_wichtigkeit(q=24):
    """Wichtigkeit je Merkmal (Mittel von |Beitrag| über die gewichteten Stützpunkte der Bevölkerung, Hintergrund dieselbe Bevölkerung). Rückgabe (5,)."""
    Xp, wp = bevoelkerung(q)
    phi = np.abs(shapley5(werte(G.wahrheit, Xp, Xp, wp)))
    return wp @ phi


def art_anteil_von(imp):
    return float(imp[0] / (imp[0] + imp[4]))
