"""Stichproben-Erklärer (Permutationsform, Štrumbelj und Kononenko 2010, Gl. 7): je Stichprobe eine Reihenfolge der Merkmale und eine Hintergrundzeile.

Der Beitrag des Merkmals j ist das Mittel der Zuwächse f(x_S ∪ j, b_rest) - f(x_S, b_rest) über M gezogene (Reihenfolge, Hintergrundzeile)-Paare, wobei S die Merkmale
vor j in der Reihenfolge sind. Kosten: M * 6 + 1 Auswertungen je Zeile (M Reihenfolgen mit je 6 Zuwächsen, die leere Koalition einmal je Hintergrundzeile; hier als Wertetabelle gelesen).
Der Zufallsstrom ist SplitMix64 (`erv_rng.SplitMix64`); `u64_block` liefert dieselben Ausgaben in einem Zug mit NumPy (Zähler-Eigenschaft von SplitMix64: der k-te Zustand ist
seed + k * Konstante), damit eine Stichprobe mit 100 Läufen zu 400 Reihenfolgen nicht Millionen Python-Aufrufe braucht. `tests/test_sampler.py` prüft Bit für Bit gegen die Klasse."""

import numpy as np

from erv_rng import MASK, SplitMix64

D = 6
_GAMMA = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def u64_block(seed, n, start=0):
    """Die Ausgaben start, start + 1, ..., start + n - 1 von SplitMix64(seed) als uint64-Feld (gleich den Aufrufen von next_u64)."""
    with np.errstate(over="ignore"):
        k = np.arange(start + 1, start + n + 1, dtype=np.uint64)
        z = np.uint64(seed & MASK) + k * _GAMMA
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        return z ^ (z >> np.uint64(31))


def mix_seed(*keys):
    """Ein Zufalls-Seed aus mehreren Zahlen (z. B. Grund-Seed, Lauf, Zeile), unabhängig von der Reihenfolge der Aufrufe: je Schlüssel ein Mischschritt."""
    s = 0x243F6A8885A308D3
    for k in keys:
        s = SplitMix64((s ^ (int(k) * 0x9E3779B97F4A7C15)) & MASK).next_u64()
    return s


def draw(seed, n_samples, n_bg):
    """n_samples Paare (Reihenfolge, Hintergrundzeile) aus dem Strom `seed`: je Paar sechs Ausgaben (fünf für das Mischen nach Fisher und Yates mit below(i + 1), eine für `below(n_bg)`).
    Rückgabe: perms (n_samples, 6) mit der Merkmalsnummer an jeder Stelle, z (n_samples,)."""
    u = u64_block(seed, 6 * n_samples).reshape(n_samples, 6)
    perms = np.tile(np.arange(D), (n_samples, 1))
    rows = np.arange(n_samples)
    for step, i in enumerate(range(D - 1, 0, -1)):
        j = (u[:, step] % np.uint64(i + 1)).astype(np.int64)
        pi, pj = perms[rows, i].copy(), perms[rows, j].copy()
        perms[rows, i], perms[rows, j] = pj, pi
    z = (u[:, 5] % np.uint64(n_bg)).astype(np.int64)
    return perms, z


def draw_reference(seed, n_samples, n_bg):
    """Langsame Referenz mit der Klasse SplitMix64 (Ausgabe wie `draw`); nur für den Test."""
    rng = SplitMix64(seed)
    perms, z = [], []
    for _ in range(n_samples):
        p = list(range(D))
        rng.shuffle(p)
        perms.append(p)
        z.append(rng.below(n_bg))
    return np.array(perms), np.array(z)


def estimate(table, perms, z):
    """Stichprobenschätzer aus der Wertetabelle einer Zeile. table (64, n_bg); perms (..., M, 6); z (..., M). Rückgabe (..., 6): Mittel der Zuwächse über die M Paare."""
    perms = np.asarray(perms)
    bits = (1 << perms).astype(np.int64)
    masks = np.concatenate([np.zeros(perms.shape[:-1] + (1,), dtype=np.int64), np.cumsum(bits, axis=-1)], axis=-1)   # (..., M, 7): bitweise disjunkt, Summe = Vereinigung
    v = table[masks, z[..., None]]
    inc = v[..., 1:] - v[..., :-1]
    out = np.zeros(perms.shape)
    np.put_along_axis(out, perms, inc, axis=-1)
    return out.mean(axis=-2)


def run_many(table, m, n_runs, seed, paired=False):
    """n_runs unabhängige Läufe des Erklärers mit m Reihenfolgen auf der Wertetabelle einer Zeile; Rückgabe (n_runs, 6).
    paired: m / 2 Reihenfolgen und ihre Umkehrung mit derselben Hintergrundzeile (m gerade)."""
    if paired and m % 2:
        raise ValueError("gepaart braucht eine gerade Zahl M")
    n_bg = table.shape[1]
    base = m // 2 if paired else m
    out = np.empty((n_runs, D))
    for r in range(n_runs):
        perms, z = draw(mix_seed(seed, r), base, n_bg)
        if paired:
            perms = np.concatenate([perms, perms[:, ::-1]])
            z = np.concatenate([z, z])
        out[r] = estimate(table, perms, z)
    return out


def cost(m):
    """Modellauswertungen je Zeile bei M Reihenfolgen: M * 6 + 1; die Aufzählung braucht 64 * 40 = 2560 (bei 40 Hintergrundzeilen)."""
    return m * D + 1


def enumeration_cost(n_bg=40):
    return (1 << D) * n_bg


def estimate_rows(tables, perms, z):
    """Wie `estimate`, aber je Zeile eine eigene Wertetabelle: tables (R, 64, n_bg); perms (R, M, 6); z (R, M). Rückgabe (R, 6)."""
    perms = np.asarray(perms)
    bits = (1 << perms).astype(np.int64)
    masks = np.concatenate([np.zeros(perms.shape[:-1] + (1,), dtype=np.int64), np.cumsum(bits, axis=-1)], axis=-1)   # (R, M, 7)
    r = np.arange(tables.shape[0])[:, None, None]
    v = tables[r, masks, z[..., None]]
    inc = v[..., 1:] - v[..., :-1]
    out = np.zeros(perms.shape)
    np.put_along_axis(out, perms, inc, axis=-1)
    return out.mean(axis=-2)


def run_rows(tables, m, seed, paired=False):
    """Ein Lauf des Erklärers über alle Zeilen einer Menge (tables (R, 64, n_bg)): m Reihenfolgen je Zeile, ein Strom `seed` für die ganze Menge. Rückgabe (R, 6)."""
    if paired and m % 2:
        raise ValueError("gepaart braucht eine gerade Zahl M")
    r, n_bg = tables.shape[0], tables.shape[2]
    base = m // 2 if paired else m
    perms, z = draw(seed, r * base, n_bg)
    perms, z = perms.reshape(r, base, D), z.reshape(r, base)
    if paired:
        perms = np.concatenate([perms, perms[:, :, ::-1]], axis=1)
        z = np.concatenate([z, z], axis=1)
    return estimate_rows(tables, perms, z)
