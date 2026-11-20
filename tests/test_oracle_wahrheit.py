"""Orakel für die Wahrheit je Funktional: exakte Bevölkerung gegen einen anderen Rechenweg (Monte-Carlo-Stichprobe der Verteilung) und gegen die eingefrorene Zahl.

„Exakte Bevölkerung“ = die analytische Rechnung (1224 gewichtete Stützpunkte, Art-Anteil 0,537); „Referenzmenge“ = eine große Stichprobe des Generators. Hier: (a) die Effizienz der
Bevölkerungs-Rechnung, (b) dieselbe Größe mit gröberer Quadratur im Band, (c) die Wichtigkeit aus einer Stichprobe als Hintergrund und als erklärte Zeilen im Band um die Bevölkerung,
(d) die eingefrorene Zahl stimmt mit der Datei der Messung."""

import json
from pathlib import Path

import numpy as np
import pytest

import erv_generator as G
import erv_wahrheit as W

DATA = Path(__file__).resolve().parent.parent / "data"


def test_effizienz_der_bevoelkerungsrechnung():
    Xp, wp = W.bevoelkerung(6)
    assert wp.sum() == pytest.approx(1.0, abs=1e-12) and len(Xp) == len(wp)
    Xe, _ = G.daten(30, 31001)
    v = W.werte(G.wahrheit, Xe, Xp, wp)
    phi = W.shapley5(v)
    assert np.max(np.abs(phi.sum(axis=1) - (v[:, -1] - v[:, 0]))) < 1e-10


def test_stuetzpunkte_zahl_und_gewichte():
    """1224 Stützpunkte bei 24 Knoten: Art 0 (12 Kunden x 2 Zoll x 2 Wochenende = 48), Art 3 (12 x 2 = 24), Export und Umschlag (je 12 x 2 x 24 = 576): 48 + 24 + 576 + 576 = 1224 (Quadratur-Zählung)."""
    Xp, wp = W.bevoelkerung(24)
    assert len(Xp) == 48 + 24 + 576 + 576 == 1224
    art_anteile = [wp[Xp[:, 0] == a].sum() for a in range(4)]
    assert art_anteile == pytest.approx(list(G.ART_P), abs=1e-12)


def test_art_anteil_mit_groeberer_quadratur_liegt_im_band_der_eingefrorenen_zahl():
    wahr = json.load(open(DATA / "erv_wahrheit.json", encoding="utf-8"))
    imp = W.exakte_wichtigkeit(6)
    assert W.art_anteil_von(imp) == pytest.approx(wahr["art_anteil_exakt"], abs=0.005)


def test_eingefrorene_exakte_zahl():
    wahr = json.load(open(DATA / "erv_wahrheit.json", encoding="utf-8"))
    assert round(wahr["art_anteil_exakt"], 3) == 0.537 and round(wahr["art_anteil_exakt_12_knoten"], 3) == 0.538          # Quadratur mit 24 bzw. 12 Knoten
    assert wahr["wichtigkeit_exakt"][5] == 0.0                       # Gewicht ist ein Nullspieler
    imp = wahr["wichtigkeit_exakt"]
    assert [round(x, 3) for x in imp[:5]] == [0.651, 0.186, 0.072, 0.055, 0.561]
    assert imp[0] / (imp[0] + imp[4]) == pytest.approx(wahr["art_anteil_exakt"], abs=1e-12)


def test_stichprobe_als_hintergrund_und_zeilen_liegt_im_band_um_die_bevoelkerung():
    """Anderer Rechenweg: Hintergrund 400 Zeilen und 400 erklärte Zeilen aus dem Generator (Stichprobe statt Quadratur) geben den Art-Anteil bis auf Stichprobenfehler."""
    Xb, _ = G.daten(400, 51001)
    Xe, _ = G.daten(400, 51002)
    wb = np.full(len(Xb), 1.0 / len(Xb))
    ph = np.abs(W.shapley5(W.werte(G.wahrheit, Xe, Xb, wb))).mean(axis=0)
    wahr = json.load(open(DATA / "erv_wahrheit.json", encoding="utf-8"))
    assert W.art_anteil_von(ph) == pytest.approx(wahr["art_anteil_exakt"], abs=0.05)
