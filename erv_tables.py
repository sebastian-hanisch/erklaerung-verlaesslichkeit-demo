"""Lesen der eingefrorenen Daten (data/): Beiträge der 160 Modelle, Wahrheit je Funktional, Wertetabellen für den Stichproben-Erklärer, Schicht Erklärer, exakte Bevölkerung.

Die Dateien erzeugt `tools/erv_make_tables.py` (mit scikit-learn im Test-venv; die App trainiert nie ein Modell). Herkunft und Stand stehen in `data/QUELLE.json`.
Feldordnung: Zelle = Klasse * 2 + Größe (Klassen gbm, gbm_ts, rf, mlp; Größen 300, 3000)."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

import erv_constants as C
import erv_generator as G

DATA = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True, eq=False)
class Bundle:
    phi: np.ndarray        # (8, 20, 200, 6) Beiträge: 20 Modelle je Zelle, 200 Zeilen (10 Zeilenmengen zu 20), 6 Merkmale
    truth: np.ndarray      # (2, 10, 200, 6) Wahrheit je Funktional je Trainingsgröße und Historie (Hintergrund: erste 40 Trainingszeilen der Historie)
    r2: np.ndarray         # (8,) mittleres R² gegen mu auf 300 frischen Zeilen
    tab: np.ndarray | None  # (8, 60, 64, 40) Wertetabellen f(x_S, b_rest): Historie 0, Modell-Zufalls-Seed 0, Zeilenmengen 1 bis 3
    erk: dict              # {Zelle: [10 Zeilenmengen: {"u30": {...}, "u100": {...}, ("p30", "p100")}]}
    wahrheit: dict         # exakte Bevölkerung


def config(klasse, n):
    return C.KLASSEN.index(klasse) * 2 + C.N_TRAIN.index(n)


@lru_cache(maxsize=1)
def load():
    d = np.load(DATA / "erv_phi.npz")
    t = np.load(DATA / "erv_tables.npz")
    with open(DATA / "erv_erklaerer.json", encoding="utf-8") as f:
        erk = json.load(f)["zellen"]
    with open(DATA / "erv_wahrheit.json", encoding="utf-8") as f:
        wahr = json.load(f)
    return Bundle(d["phi"].astype(float), d["truth"].astype(float), d["r2"].astype(float), t["tab"].astype(float), {int(k): v for k, v in erk.items()}, wahr)


@lru_cache(maxsize=1)
def zeilen():
    """Die 200 erklärten Zeilen (10 Zeilenmengen zu 20): Daten-Zufalls-Seeds 41001 bis 41010 des Generators."""
    s0 = C.SEEDSETS["erst"]["zeilen0"]
    return np.vstack([G.daten(C.N_ZEILEN, s0 + s)[0] for s in range(C.N_MENGEN)])


def zeile_text(i):
    """Die Merkmale der Zeile i (0 bis 199) als kurzer Text."""
    x = zeilen()[i]
    return (f"Art {C.ART_NAMEN[int(x[0])]}, Kunde {int(x[1]) + 1}, Zoll {'ja' if x[2] else 'nein'}, Wochenende {'ja' if x[3] else 'nein'}, "
            f"Vorlauf {x[4]:.1f} Tage, Gewicht {x[5]:.2f}".replace(".", ","))


@lru_cache(maxsize=1)
def load_zweit():
    """Zweite Messung mit anderen Historien (Daten-Zufalls-Seeds 300 bis 309) und anderen Zeilenmengen (43001 bis 43010); ohne Wertetabellen (nur Beiträge, Wahrheit je Funktional, Schicht Erklärer)."""
    d = np.load(DATA / "erv_phi_zweit.npz")
    with open(DATA / "erv_erklaerer_zweit.json", encoding="utf-8") as f:
        erk = json.load(f)["zellen"]
    with open(DATA / "erv_wahrheit.json", encoding="utf-8") as f:
        wahr = json.load(f)
    return Bundle(d["phi"].astype(float), d["truth"].astype(float), d["r2"].astype(float), None, {int(k): v for k, v in erk.items()}, wahr)
