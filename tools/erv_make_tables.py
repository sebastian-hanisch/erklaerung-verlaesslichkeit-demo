"""Erzeugt die eingefrorenen Daten der Demo (data/erv_*.npz und data/erv_*.json) oder prüft sie gegen eine Neurechnung.

Die App trainiert nie ein Modell: sklearn (nur hier und in requirements-dev.txt) trainiert die 160 Modelle (4 Klassen x 2 Trainingsgrößen x 20 Modelle: 10 Historien mit
Modell-Zufalls-Seed 0 und die Historien 0 bis 4 mit den Zufalls-Seeds 1 und 2), und die exakte Aufzählung über 64 Koalitionen x 40 Hintergrundzeilen läuft einmal je Modell und
Zeile. Die Stichproben des Erklärers lesen aus denselben Wertetabellen (SplitMix64-Strom, plattformfest).

Aufruf (Test-venv mit scikit-learn, im Ordner der Demo):
  python tools/erv_make_tables.py --write [--satz erst|zweit]   schreibt die Dateien (Zeit: einige Minuten je Satz)
  python tools/erv_make_tables.py --check [--teil klein|voll]    rechnet neu und vergleicht (klein: ein Modell je Klasse, Historie 0, n = 300)
Die Datei data/erv_zweit.json (Wiederholung mit anderen Historien und Zeilenmengen) steht nur in tools-Läufen und der README, nicht in der App."""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import erv_constants as C  # noqa: E402
import erv_exact as E  # noqa: E402
import erv_generator as G  # noqa: E402
import erv_schichten as SC  # noqa: E402
import erv_wahrheit as W  # noqa: E402

warnings.filterwarnings("ignore")
DATA = ROOT / "data"


def mache_modell(klasse, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    if klasse == "gbm":
        return GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, learning_rate=0.1)
    if klasse == "gbm_ts":
        return GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.7, max_features=0.7)
    if klasse == "rf":
        return RandomForestRegressor(random_state=seed, n_estimators=60, min_samples_leaf=3, max_features=0.5, n_jobs=1)
    if klasse == "mlp":
        ct = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False), [0, 1]), ("sk", StandardScaler(), [2, 3, 4, 5])])
        return make_pipeline(ct, MLPRegressor(random_state=seed, hidden_layer_sizes=(16,), alpha=3.0, max_iter=500, solver="lbfgs"))
    raise ValueError(klasse)


def zeilenmengen(satz):
    """Die zehn Zeilenmengen zu je 20 Zeilen (Daten-Zufalls-Seeds zeilen0 bis zeilen0 + 9), als (200, 6)."""
    s0 = C.SEEDSETS[satz]["zeilen0"]
    return np.vstack([G.daten(C.N_ZEILEN, s0 + s)[0] for s in range(C.N_MENGEN)])


def rechne(satz, klassen=C.KLASSEN, groessen=C.N_TRAIN, hist_max=C.N_MENGEN, mit_erklaerer=True, log=print):
    """Rechnet alle Daten eines Seed-Satzes. Rückgabe ein Wörterbuch mit phi (8, 20, 200, 6), truth (2, 10, 200, 6), r2 (8,), tab (8, 60, 64, 40), erk (JSON-fähig)."""
    cfg = C.SEEDSETS[satz]
    rows = zeilenmengen(satz)
    xr, _ = G.daten(C.N_REF_ZEILEN, cfg["ref"])
    mu_r = G.wahrheit(xr)
    nconf = C.N_CONFIG
    phi = np.zeros((nconf, 20, len(rows), 6), dtype=np.float32)
    truth = np.zeros((2, C.N_MENGEN, len(rows), 6), dtype=np.float32)
    r2 = np.zeros(nconf)
    tab = np.zeros((nconf, C.N_TAB_MENGEN * C.N_ZEILEN, 64, C.N_HINTERGRUND), dtype=np.float32)
    erk = {}
    t0 = time.time()
    for ni, n in enumerate(C.N_TRAIN):
        if n not in groessen:
            continue
        for h in range(hist_max):
            X, y = G.daten(n, cfg["hist0"] + h)
            B = X[:C.N_HINTERGRUND]
            truth[ni, h] = E.exact_from_table(E.coalition_values(G.wahrheit, rows, B, chunk=200000)).astype(np.float32)
    for ci, kl in enumerate(C.KLASSEN):
        if kl not in klassen:
            continue
        for ni, n in enumerate(C.N_TRAIN):
            if n not in groessen:
                continue
            conf = ci * 2 + ni
            r2l = []
            tabs_erk = {}
            for h in range(hist_max):
                X, y = G.daten(n, cfg["hist0"] + h)
                B = X[:C.N_HINTERGRUND]
                for s in C.MODELL_SEEDS if h < SC.N_MODELL_HIST else (0,):
                    m = mache_modell(kl, s).fit(X, y)
                    T = E.coalition_values(m.predict, rows, B, chunk=200000)
                    phi[conf, SC.modell_index(h, s)] = E.exact_from_table(T).astype(np.float32)
                    if s == 0:
                        pr = m.predict(xr)
                        r2l.append(1 - float(np.sum((pr - mu_r) ** 2) / np.sum((mu_r - mu_r.mean()) ** 2)))
                        if h < C.N_ERK_HIST:
                            tabs_erk[h] = T
                        if h == 0:
                            tab[conf] = T[:C.N_TAB_MENGEN * C.N_ZEILEN].astype(np.float32)
            r2[conf] = float(np.mean(r2l))
            if mit_erklaerer:
                erk[str(conf)] = erklaerer_zellen(satz, conf, kl, phi[conf], tabs_erk)
            log(f"  {kl} n = {n}: R2 {r2[conf]:.2f} [{time.time() - t0:.0f} s]")
    return {"phi": phi, "truth": truth, "r2": r2, "tab": tab, "erk": erk}


def erklaerer_zellen(satz, conf, kl, phi_conf, tabs):
    """Schicht Erklärer je Zeilenmenge (10): M = 30 und 100 ungepaart, bei Boosting und Netz zusätzlich gepaart; Mittel über die Historien 0 bis 3, je 10 Läufe.
    Die Aufzählung (exakte Beiträge) kommt aus derselben Wertetabelle wie der Stichprobenschätzer (nicht aus phi, das in float32 steht)."""
    cfg = C.SEEDSETS[satz]
    erg = []
    for menge in range(C.N_MENGEN):
        sl = SC.set_slice(menge)
        zelle = {}
        for variante in ("u", "p") if kl in ("gbm", "mlp") else ("u",):
            for m in SC.M_LISTE:
                werte = []
                for h in range(C.N_ERK_HIST):
                    T = tabs[h][sl]
                    ex = E.exact_from_table(T)
                    seed = cfg["erk"] * 1_000_003 + conf * 10_007 + menge * 131 + h * 17 + m + (7 if variante == "p" else 0)
                    werte.append(SC.erklaerer_zelle(T, ex, m, C.N_LAEUFE_FEST, seed, paired=(variante == "p")))
                zelle[f"{variante}{m}"] = {k: float(np.mean([w[k] for w in werte])) for k in werte[0]}
        erg.append(zelle)
    return erg


def schreibe(d, satz):
    DATA.mkdir(exist_ok=True)
    if satz == "erst":
        np.savez_compressed(DATA / "erv_phi.npz", phi=d["phi"], truth=d["truth"], r2=d["r2"].astype(np.float32))
        np.savez_compressed(DATA / "erv_tables.npz", tab=d["tab"])
        json.dump({"satz": satz, "zellen": d["erk"]}, open(DATA / "erv_erklaerer.json", "w", encoding="utf-8"), indent=0)
    else:
        np.savez_compressed(DATA / "erv_phi_zweit.npz", phi=d["phi"], truth=d["truth"], r2=d["r2"].astype(np.float32))
        json.dump({"satz": satz, "zellen": d["erk"]}, open(DATA / "erv_erklaerer_zweit.json", "w", encoding="utf-8"), indent=0)


def schreibe_wahrheit():
    imp = W.exakte_wichtigkeit(24)
    imp12 = W.exakte_wichtigkeit(12)
    out = {"art_anteil_exakt": W.art_anteil_von(imp), "art_anteil_exakt_12_knoten": W.art_anteil_von(imp12), "wichtigkeit_exakt": [float(v) for v in imp] + [0.0],
           "quelle": "erv_wahrheit.exakte_wichtigkeit(24): 1224 gewichtete Stützpunkte, Hintergrund und erklärte Zeilen dieselbe Bevölkerung, Gauß-Legendre 24 Knoten über den Vorlauf"}
    json.dump(out, open(DATA / "erv_wahrheit.json", "w", encoding="utf-8"), indent=1)
    return out


def pruefe(satz_teil):
    """Rechnet neu und vergleicht mit den Dateien. klein: ein Modell je Klasse (n = 300, Historie 0, Modell-Zufalls-Seed 0) und die Wahrheit; voll: alles."""
    alt = np.load(DATA / "erv_phi.npz")
    schlecht = []
    if satz_teil == "klein":
        rows = zeilenmengen("erst")
        X, y = G.daten(300, C.SEEDSETS["erst"]["hist0"])
        B = X[:C.N_HINTERGRUND]
        for ci, kl in enumerate(C.KLASSEN):
            m = mache_modell(kl, 0).fit(X, y)
            neu = E.exact_from_table(E.coalition_values(m.predict, rows, B, chunk=200000))
            abw = float(np.max(np.abs(neu - alt["phi"][ci * 2, 0])))
            print(f"{kl}: größte Abweichung der Beiträge gegen die Datei {abw:.2e}")
            schlecht.append((kl, abw))
        tw = E.exact_from_table(E.coalition_values(G.wahrheit, rows, B, chunk=200000))
        abw = float(np.max(np.abs(tw - alt["truth"][0, 0])))
        print(f"Wahrheit: {abw:.2e}")
        schlecht.append(("wahrheit", abw))
        return schlecht
    d = rechne("erst", log=print)
    for name in ("phi", "truth"):
        abw = float(np.max(np.abs(d[name] - alt[name])))
        print(f"{name}: größte Abweichung {abw:.2e}")
        schlecht.append((name, abw))
    return schlecht


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--satz", default="erst", choices=("erst", "zweit"))
    ap.add_argument("--teil", default="klein", choices=("klein", "voll"))
    ap.add_argument("--wahrheit", action="store_true", help="nur die exakte Bevölkerung neu rechnen und schreiben (20 s)")
    a = ap.parse_args()
    if a.wahrheit:
        print(schreibe_wahrheit())
        return
    if a.write:
        t0 = time.time()
        d = rechne(a.satz)
        schreibe(d, a.satz)
        print(f"fertig nach {time.time() - t0:.0f} s")
    if a.check:
        pruefe(a.teil)


if __name__ == "__main__":
    main()
