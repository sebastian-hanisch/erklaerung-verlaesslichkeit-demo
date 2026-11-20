"""Exakte Aufzählung: interventionaler Shapley-Wert über alle 2^6 Koalitionen (Koalitionsformel) aus einer Wertetabelle.

Wertfunktional (interventional): v(S) = Mittel über die Hintergrundzeilen b von f(x_S, b_rest), also die Merkmale in S von der erklärten Zeile x,
die übrigen von einer Hintergrundzeile. Tabelle T[zeile, S, b] = f(x_S, b_rest) mit S als Bitmaske (Bit j gesetzt = Merkmal j kommt von x).
Die Wertetabelle ist das, was das Modell nur einmal gefragt wird; Aufzählung und Stichprobe lesen aus ihr (Stichprobe: Tabellenabruf)."""

from math import factorial

import numpy as np

D = 6
K = 1 << D
MASKS = np.array([[(k >> j) & 1 for j in range(D)] for k in range(K)], dtype=bool)
SIZE = MASKS.sum(axis=1)
WEIGHT = np.array([factorial(s) * factorial(D - s - 1) / factorial(D) if s < D else 0.0 for s in range(D + 1)])
# je Merkmal j: alle Koalitionen ohne j, dieselben mit j, Gewicht |S|! (d - |S| - 1)! / d!
_IDX = []
for _j in range(D):
    _ks = np.array([k for k in range(K) if not (k >> _j) & 1])
    _IDX.append((_ks, _ks | (1 << _j), WEIGHT[SIZE[_ks]]))


def shapley_from_values(v):
    """v (..., 64) Koalitionswerte -> Shapley-Werte (..., 6) nach der Koalitionsformel."""
    v = np.asarray(v, dtype=float)
    phi = np.empty(v.shape[:-1] + (D,))
    for j, (ks, kj, w) in enumerate(_IDX):
        phi[..., j] = ((v[..., kj] - v[..., ks]) * w).sum(axis=-1)
    return phi


def coalition_values(f, rows, background, chunk=60000):
    """Tabelle T (n_zeilen, 64, n_hintergrund) mit T[i, S, b] = f(x_i auf S, b_rest); f erhält (m, 6)-Matrizen und liefert (m,)."""
    rows = np.asarray(rows, dtype=float)
    background = np.asarray(background, dtype=float)
    nb = len(background)
    out = np.empty((len(rows), K, nb))
    per = K * nb
    step = max(1, chunk // per)
    for a in range(0, len(rows), step):
        xs = rows[a:a + step]
        H = np.where(MASKS[None, :, None, :], xs[:, None, None, :], background[None, None, :, :])
        out[a:a + step] = f(H.reshape(-1, D)).reshape(len(xs), K, nb)
    return out


def exact_from_table(T):
    """Exakte Beiträge (n_zeilen, 6) aus der Wertetabelle: Mittel über den Hintergrund, dann Koalitionsformel."""
    return shapley_from_values(np.asarray(T, dtype=float).mean(axis=-1))


def efficiency_gap(T):
    """Größter Abstand von Summe der Beiträge und f(x) - Mittel von f über den Hintergrund (Effizienz der Aufzählung); v(leer) = v(voll aus Hintergrund)."""
    T = np.asarray(T, dtype=float)
    v = T.mean(axis=-1)
    return float(np.max(np.abs(shapley_from_values(v).sum(axis=-1) - (v[..., K - 1] - v[..., 0]))))
