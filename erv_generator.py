"""Erfundener Generator für Container-Verweilzeiten (Annahmen, kein Echtdatenbezug) mit bekannter Wahrheit.

KOPIE (nicht Import) der Funktionen `kunden_effekte`, `log_median_and_s`, `draw_features` und `make_history` aus `verweilzeit-demo/vwz_generator.py` mit den
Konstanten aus `vwz_constants.py`; die Rechenschritte und die Reihenfolge der Zufallszahlen sind unverändert, `tests/test_copies.py` pinnt feste Werte der Quelle.
Nicht um Docstring-Korrekturen erweitern: eine Textänderung bräche den Kopie-Test (DEMO-PLAYBOOK 11a Punkt 11).

Zielgröße ist ln(Verweilzeit) (`daten`). Die wahre bedingte Mitte mu(x) (`wahrheit`) ist analytisch bekannt und dient als Wahrheit je Funktional.
Spalten: 0 art (0 Import, 1 Export, 2 Umschlag, 3 Leer), 1 kunde (0 bis 11), 2 zoll, 3 wochenende, 4 vorlauf, 5 gewicht (reines Rauschen)."""

import numpy as np

# Konstanten (verweilzeit-demo/vwz_constants.py)
ART_P = (0.35, 0.30, 0.25, 0.10)
K_KUNDEN = 12
KUNDEN_SEED = 12345
U_SD = 0.30
U_SCALE = (1.0, 0.3, 0.3, 1.0)
S_BASE = (0.55, 0.25, 0.35, 0.90)
P_ZOLL = 0.25
HISTORY_SEED_OFFSET = 10_000_000
COL_ANNOUNCED = 6

NAMEN = ("Art", "Kunde", "Zoll", "Wochenende", "Vorlauf", "Gewicht")
D = 6


def kunden_effekte(k_seed=KUNDEN_SEED):
    return np.random.default_rng(k_seed).normal(0.0, U_SD, K_KUNDEN)


UK = kunden_effekte()


def log_median_and_s(art, kunde, zoll, we, vorlauf, noise_scale=1.0):
    """Analytische bedingte Verteilung: log-Verweilzeit ~ N(mu, s^2). Vektorisiert."""
    art = np.asarray(art)
    u = UK[kunde] * np.take(U_SCALE, art)
    med = np.select(
        [art == 0, art == 1, art == 2],
        [3.5 * np.exp(0.7 * zoll + 0.30 * we), 0.6 + 0.85 * vorlauf, 0.3 + 0.8 * vorlauf],
        default=7.0 * np.exp(0.30 * we),
    )
    mu = np.log(med) + u
    s = np.take(S_BASE, art) * noise_scale
    return mu, s


def draw_features(rng, n):
    """Nur für Trainingsdaten: unabhängige Historie ohne Blockbeschränkung (Ankunftszeit gleichverteilt über viele Wochen)."""
    art = rng.choice(4, size=n, p=ART_P)
    kunde = rng.integers(0, K_KUNDEN, n)
    zoll = ((rng.random(n) < P_ZOLL) & (art == 0)).astype(int)
    tday = rng.random(n) * 7000.0
    we = ((np.floor(tday).astype(int) % 7) >= 5).astype(int)
    vorlauf = np.where(art == 1, rng.uniform(1, 10, n), np.where(art == 2, rng.uniform(0.5, 6, n), 0.0))
    gewicht = rng.normal(0, 1, n)
    return np.column_stack([art, kunde, zoll, we, vorlauf, gewicht, np.zeros(n)]).astype(float)


def make_history(n, seed, noise_scale=1.0, nu=None):
    """Trainingsdaten: n abgeschlossene Container aus einer eigenen Historie (eigener Seed-Bereich): X, Verweilzeit."""
    rng = np.random.default_rng(HISTORY_SEED_OFFSET + seed)
    X = draw_features(rng, n)
    mu, s = log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4], noise_scale)
    z = rng.standard_normal(n)
    za = rng.standard_normal(n)
    d = np.exp(mu + s * z)
    if nu is not None:
        X[:, COL_ANNOUNCED] = d * np.exp(nu * za)
    return X, d


def daten(n, seed):
    """(X (n, 6), y = ln Verweilzeit): sechs Merkmale ohne die Ankündigungsspalte."""
    X, d = make_history(n, seed)
    return X[:, :D].copy(), np.log(d)


def wahrheit(X):
    """Wahre bedingte Mitte mu(x) = ln Median der Verweilzeit zu den Merkmalen X (n, 6)."""
    mu, _ = log_median_and_s(X[:, 0].astype(int), X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4])
    return mu
