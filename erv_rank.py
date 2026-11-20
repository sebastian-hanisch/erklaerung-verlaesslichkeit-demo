"""Rangmaß und Betragsmaß (beide werden gezeigt): Rangkorrelation mit mittleren Rängen bei Gleichstand, relativer L2-Abstand, Rang-1-Merkmal.

Die Funktionen arbeiten auf der letzten Achse (Merkmale) und beliebig vielen davor, damit ganze Paarlisten ohne Schleife gerechnet werden."""

import numpy as np


def mean_ranks(a):
    """Mittlere Ränge (1 = kleinster Wert) auf der letzten Achse; Gleichstände teilen sich den Mittelwert ihrer Ränge."""
    a = np.asarray(a, dtype=float)
    less = (a[..., None, :] < a[..., :, None]).sum(axis=-1)
    equal = (a[..., None, :] == a[..., :, None]).sum(axis=-1)
    return less + (equal + 1) / 2.0


def spearman(a, b):
    """Spearman-Rangkorrelation auf der letzten Achse; nan, wenn eine Seite konstant ist (undefiniert, wird gezählt, nicht als 0 gerechnet)."""
    ra, rb = mean_ranks(a), mean_ranks(b)
    ra = ra - ra.mean(axis=-1, keepdims=True)
    rb = rb - rb.mean(axis=-1, keepdims=True)
    num = (ra * rb).sum(axis=-1)
    den = np.sqrt((ra * ra).sum(axis=-1) * (rb * rb).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def rel_l2(est, ref):
    """Relativer L2-Abstand ||est - ref|| / ||ref|| auf der letzten Achse (nan bei ||ref|| = 0)."""
    est, ref = np.asarray(est, dtype=float), np.asarray(ref, dtype=float)
    nr = np.linalg.norm(ref, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nr > 0, np.linalg.norm(est - ref, axis=-1) / np.where(nr > 0, nr, 1.0), np.nan)


def rang1(a):
    """Index des größten Werts auf der letzten Achse; bei Gleichstand der kleinste Index (feste Aufzählungsreihenfolge der Merkmale)."""
    return np.argmax(np.asarray(a, dtype=float), axis=-1)


def wichtigkeit(phi):
    """Globale Wichtigkeit: Mittel von |Beitrag| über die Zeilen (vorletzte Achse)."""
    return np.abs(np.asarray(phi, dtype=float)).mean(axis=-2)


def art_anteil(imp):
    """Anteil der Art an (Art + Vorlauf) der Wichtigkeit (Merkmale 0 und 4)."""
    imp = np.asarray(imp, dtype=float)
    nenner = imp[..., 0] + imp[..., 4]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(nenner > 0, imp[..., 0] / np.where(nenner > 0, nenner, 1.0), np.nan)


def nanmean(a, axis=None):
    """Mittel ohne nan-Werte; nan (ohne Warnung), wenn alle fehlen. Undefinierte Rangkorrelationen werden ausgelassen und getrennt gezählt, nicht als 0 gerechnet."""
    a = np.asarray(a, dtype=float)
    ok = np.isfinite(a)
    n = ok.sum(axis=axis)
    tot = np.where(ok, a, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, tot / np.where(n > 0, n, 1), np.nan)


def pair_indices(n):
    """Alle Paare (a, b) mit a < b aus n Elementen, als zwei Indexfelder."""
    a, b = np.triu_indices(n, k=1)
    return a, b


def paar_masse(phis):
    """phis: (n_modelle, n_zeilen, 6) Beiträge derselben Zeilen aus n Modellen. Liefert je Paar von Modellen vier Zahlen:
    Rangkorrelation der Beiträge (Mittel über Zeilen, nan-Zeilen ausgelassen), Anteil Zeilen mit gleichem Rang 1, Rangkorrelation der Wichtigkeit, Rang 1 der Wichtigkeit gleich (0/1).
    Rückgabe: Feld (n_paare, 4)."""
    phis = np.abs(np.asarray(phis, dtype=float))
    a, b = pair_indices(len(phis))
    rho = spearman(phis[a], phis[b])                      # (paare, zeilen)
    z = nanmean(rho, axis=1)
    gl1 = (rang1(phis[a]) == rang1(phis[b])).mean(axis=1)
    imp = phis.mean(axis=1)
    rw = spearman(imp[a], imp[b])
    gw = (rang1(imp[a]) == rang1(imp[b])).astype(float)
    return np.stack([z, gl1, rw, gw], axis=1)
