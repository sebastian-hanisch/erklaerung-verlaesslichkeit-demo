"""Die drei Schichten des Zufalls aus den eingefrorenen Beiträgen (Daten: Neu-Training, Modell: Zufalls-Seed des Modells, Erklärer: Lauf der Stichprobe) und die Zeilenmenge.

Alle Funktionen sind reine Rechnungen auf Feldern; die Daten stehen in `data/` (siehe `erv_tables.py`), erzeugt von `tools/erv_make_tables.py`.
Zähleinheiten: Daten und Modell zählen Paare von Neu-Trainings bzw. Zufalls-Seeds (45 Paare aus 10 Neu-Trainings), der Erklärer Paare von Läufen (45 Paare aus 10 Läufen),
alles je Zeilenmenge (20 Zeilen); die Spanne ist die über zehn getrennte Zeilenmengen."""

import numpy as np

import erv_rank as R
import erv_sampler as S

N_SET, N_ROWS = 10, 20            # zehn Zeilenmengen zu je 20 Zeilen
N_DATEN, N_MODELL_HIST = 10, 5    # Neu-Trainings (Historien); Historien mit zusätzlichen Modell-Zufalls-Seeds
M_LISTE = (30, 100)


def set_slice(menge):
    """Zeilenindizes einer Zeilenmenge (0 bis 9) oder aller 200 (menge = None)."""
    return slice(0, N_SET * N_ROWS) if menge is None else slice(menge * N_ROWS, (menge + 1) * N_ROWS)


def modell_index(hist, seed=0):
    """Position eines Modells im Feld phi (20 Modelle): 0 bis 9 = Historien mit Modell-Zufalls-Seed 0; 10 bis 19 = Historien 0 bis 4 mit den Zufalls-Seeds 1 und 2."""
    return hist if seed == 0 else N_DATEN + 2 * hist + (seed - 1)


def daten_schicht(phi, menge):
    """phi (20, 200, 6) eines Modelltyps: Paare der 10 Neu-Trainings (Modell-Zufalls-Seed 0) auf einer Zeilenmenge. Rückgabe (45, 4): Rangkorrelation Beitrag, gleiches Rang 1 (Zeilen), Rangkorrelation Wichtigkeit, gleiches Rang 1 der Wichtigkeit."""
    return R.paar_masse(phi[:N_DATEN, set_slice(menge)])


def modell_schicht(phi, menge):
    """Paare der drei Modell-Zufalls-Seeds je Historie (Historien 0 bis 4: 3 Paare je Historie, 15 Paare). Rückgabe (15, 4) wie `daten_schicht`."""
    erg = []
    for h in range(N_MODELL_HIST):
        sel = [modell_index(h, 0), modell_index(h, 1), modell_index(h, 2)]
        erg.append(R.paar_masse(phi[sel][:, set_slice(menge)]))
    return np.concatenate(erg)


def mittel(m):
    """Mittel über die Paare (nan-Werte ausgelassen)."""
    return R.nanmean(m, axis=0)


def erklaerer_zelle(tables, exact, m, n_runs, seed, paired=False):
    """Eine Zelle der Schicht Erklärer: ein Modell, eine Zeilenmenge (tables (20, 64, 40), exact (20, 6)), m Reihenfolgen, n_runs Läufe.
    Rückgabe ein Wörterbuch: l2 (Median über Läufe und Zeilen des relativen L2 zur Aufzählung), r1 (Anteil Läufe x Zeilen mit demselben Rang-1-Merkmal wie die Aufzählung),
    rk (Rangkorrelation zwischen Läufen, Mittel über Laufpaare und Zeilen), und dasselbe für die Wichtigkeit: l2w, r1w, rkw."""
    runs = np.stack([S.run_rows(tables, m, S.mix_seed(seed, r), paired) for r in range(n_runs)])      # (läufe, zeilen, 6)
    ex = np.asarray(exact, dtype=float)
    l2 = float(np.median(R.rel_l2(runs, ex[None])))
    r1 = float((R.rang1(np.abs(runs)) == R.rang1(np.abs(ex))[None]).mean())
    pm = R.paar_masse(runs)                                                                           # Paare von Läufen
    imp_ex = np.abs(ex).mean(axis=0)
    imp = np.abs(runs).mean(axis=1)
    l2w = float(np.median(R.rel_l2(imp, imp_ex[None])))
    r1w = float((R.rang1(imp) == R.rang1(imp_ex)).mean())
    mw = mittel(pm)
    return {"l2": l2, "r1": r1, "rk": float(mw[0]), "l2w": l2w, "r1w": r1w, "rkw": float(mw[2])}


def art_anteile(phi, menge):
    """Anteil Art / (Art + Vorlauf) der Wichtigkeit je Neu-Training auf einer Zeilenmenge; phi (20, 200, 6) -> (10,)."""
    return R.art_anteil(R.wichtigkeit(phi[:N_DATEN, set_slice(menge)]))


def spitzen(phi, menge):
    """Rang-1-Merkmal der Wichtigkeit je Neu-Training (10,) auf einer Zeilenmenge."""
    return R.rang1(R.wichtigkeit(phi[:N_DATEN, set_slice(menge)]))


def einheitlich(phi, menge):
    """True, wenn alle zehn Neu-Trainings dasselbe Spitzenmerkmal der Wichtigkeit nennen."""
    s = spitzen(phi, menge)
    return bool((s == s[0]).all())


def wichtigkeit_gegen_wahrheit(phi, truth, menge):
    """Rangkorrelation der Wichtigkeit jedes Neu-Trainings mit der Wahrheit je Funktional (derselbe Hintergrund, dieselbe Menge); Mittel über die 10 Neu-Trainings.
    truth (10, 200, 6): je Historie die Wahrheit mit dem Hintergrund dieser Historie."""
    imp = R.wichtigkeit(phi[:N_DATEN, set_slice(menge)])
    imp_t = R.wichtigkeit(truth[:, set_slice(menge)])
    return float(R.nanmean(R.spearman(imp, imp_t)))
