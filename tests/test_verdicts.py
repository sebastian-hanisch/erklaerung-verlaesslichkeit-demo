"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Schwelle und jede Zahl im Text steht hier. Ein Verdikt ist eine reine Funktion (Art, Text)."""

import numpy as np
import pytest

import erv_evaluation as ev


def test_zahlenformat_mit_dezimalkomma():
    assert ev.fmt(0.4449) == "0,44" and ev.fmt(0.1234, 3) == "0,123" and ev.pct(0.27) == "27 %"
    assert ev.pct(0.0) == "0 %" and ev.pct(1.0) == "100 %" and ev.pct(0.07) == "7 %"


def test_neutraining_schwellen_und_zahlen():
    sp = (0.44, 0.37, 0.54)
    wsp = (0.74, 0.58, 0.88)
    kind, text = ev.verdict_neutraining("Random Forest", 300, sp, wsp, 45)
    assert kind == "warning"
    assert "Random Forest, 300 Trainingszeilen" in text and "**0,44**" in text and "(0,37 bis 0,54 über zehn Zeilenmengen zu 20 Zeilen; 45 Paare je Menge)" in text
    assert "**0,74**" in text and "(0,58 bis 0,88)" in text
    assert ev.verdict_neutraining("x", 300, (0.69999, 0, 1), wsp, 45)[0] == "warning"      # genau unter 0,7
    assert ev.verdict_neutraining("x", 300, (0.7, 0, 1), wsp, 45)[0] == "info"           # 0,7 ist inklusive Info
    assert ev.verdict_neutraining("x", 300, (0.89999, 0, 1), wsp, 45)[0] == "info"
    assert ev.verdict_neutraining("x", 300, (0.9, 0, 1), wsp, 45)[0] == "success"        # 0,9 ist inklusive Erfolg
    assert "3 Paare je Menge" in ev.verdict_neutraining("x", 3000, sp, wsp, 3)[1] and "3000 Trainingszeilen" in ev.verdict_neutraining("x", 3000, sp, wsp, 3)[1]


def test_erklaerer_verdikt_warnt_bei_anderem_rang1_auch_bei_kleinem_abstand():
    kind, text = ev.verdict_erklaerer(100, 601, 2560, 0.148, 0.27, 30)
    assert kind == "warning"
    assert "**100** Reihenfolgen (601 von 2560 Auswertungen je Zeile)" in text and "**0,148**" in text and "**27 %** der 30 Läufe" in text
    assert "Ein kleiner Abstand schließt ein anderes Rang-1-Merkmal nicht aus" in text
    kind, text = ev.verdict_erklaerer(30, 181, 2560, 0.083, 0.0, 10)
    assert kind == "success" and "**0 %** der 10 Läufe" in text and "andere Zeilen können abweichen" in text
    assert ev.verdict_erklaerer(30, 181, 2560, 0.5, 0.0001, 10)[0] == "warning"        # schon ein Bruchteil eines Prozents ist ein anderes Rang 1


def test_paarung_und_schichten_nennen_die_zaehlung():
    kind, text = ev.verdict_paarung(80, 80, 79)
    assert kind == "info" and "**80 von 80**" in text and "**79 von 80**" in text and "nicht für gepaartes KernelSHAP" in text
    z = {"n": 80, "beitrag_kleinste_100": 80, "beitrag_kleiner_als_30": 78, "wichtigkeit_kleinste_100": 78, "wichtigkeit_kleiner_als_30": 77}
    kind, text = ev.verdict_schichten(z)
    assert kind == "info" and "**80** Zellen (4 Modellklassen x 2 Trainingsgrößen x 10 Zeilenmengen)" in text
    assert "in **80** Zellen die Schicht" in text and "**78**" in text and "**78** bzw. **77** von 80" in text


def test_stabil_verdikt_warnt_nur_bei_daneben():
    kind, text = ev.verdict_stabil("Boosting", 300, (0.31, 0.21, 0.44), 0.54, 0.537, 7, "Vorlauf", 10)
    assert kind == "warning" and "**7 von 10**" in text and "alle 10 Neu-Trainings" in text and "am häufigsten: Vorlauf" in text
    assert "**0,31** (0,21 bis 0,44 je Zeilenmenge) gegen 0,54" in text and "exakte Bevölkerung: 0,537" in text and "Stabil heißt hier nicht richtig" in text
    assert ev.verdict_stabil("Netz", 300, (0.52, 0.49, 0.59), 0.54, 0.537, 2, "Art", 10)[0] == "info"
    assert ev.verdict_stabil("x", 300, (0.43, 0, 1), 0.54, 0.537, 2, "Art", 10)[0] == "warning"        # 0,54 - 0,43 = 0,11: mehr als 0,1 daneben
    assert ev.verdict_stabil("x", 300, (0.45, 0, 1), 0.54, 0.537, 2, "Art", 10)[0] == "info"           # 0,54 - 0,45 = 0,09: nicht daneben


def test_menge_verdikt():
    kind, text = ev.verdict_menge("Zeilenmenge 3", 0.44, 0.84, (0.47, 0.38, 0.53), False)
    assert kind == "info" and "**Zeilenmenge 3**" in text and "**0,44**" in text and "**0,84**" in text and "von 0,38 bis 0,53" in text
    assert "(Die Balken zeigen die Wichtigkeit dieser Menge.)" in ev.verdict_menge("alle 200 Zeilen", 0.4, 0.8, (0.5, 0.4, 0.6), True)[1]


def test_wahrheit_rang1_und_menge_text():
    kind, text = ev.verdict_wahrheit_rang1(6)
    assert kind == "info" and "**6 von 10**" in text and "keine Kennzahl" in text
    assert ev.menge_text(ev.ALLE) == "alle 200 Zeilen" and ev.menge_text(0) == "Zeilenmenge 1" and ev.menge_text(9) == "Zeilenmenge 10"


def test_zeilenindex_und_spanne():
    assert ev.zeilenindex(ev.Settings(menge=0, zeile=1)) == 0 and ev.zeilenindex(ev.Settings(menge=2, zeile=5)) == 44
    assert ev.zeilenindex(ev.Settings(menge=ev.ALLE, zeile=7)) == 6                 # bei "alle 200" zählt die Zeile in Menge 1
    mittel, lo, hi = ev.spanne([1.0, 2.0, 3.0, np.nan] + [2.0] * 6 + [9.0])          # nur die ersten zehn Zeilenmengen zählen, nan wird ausgelassen
    assert (lo, hi) == (1.0, 3.0) and mittel == pytest.approx((1 + 2 + 3 + 12) / 9, abs=1e-12)


def test_zweit_texte_nennen_beide_zaehlungen():
    z = {"schichten_erst": {"n": 80, "beitrag_kleinste_100": 80, "beitrag_kleiner_als_30": 78, "wichtigkeit_kleinste_100": 78, "wichtigkeit_kleiner_als_30": 78},
         "schichten_zweit": {"n": 80, "beitrag_kleinste_100": 80, "beitrag_kleiner_als_30": 77, "wichtigkeit_kleinste_100": 74, "wichtigkeit_kleiner_als_30": 71},
         "einheitlich_erst": np.array([7, 9, 7, 9, 7, 8, 2, 2]), "einheitlich_zweit": np.array([5, 7, 6, 8, 7, 7, 3, 9]),
         "art_vorn_erst": np.array([6, 4, 6, 4, 6, 4, 6, 4]), "art_vorn_zweit": np.array([9, 8, 9, 8, 9, 8, 9, 8])}
    t3 = ev.text_zweit_schichten(z)
    assert "in 80 statt 80 von 80 Zellen" in t3 and "77 statt 78" in t3 and "74 statt 78" in t3 and "71 statt 78" in t3
    t4 = ev.text_zweit_spitze(z)
    assert "in 5 bis 8 von 10 Zeilenmengen (erste Messung: 7 bis 9)" in t4 and "beim Netz in 3 (300 Zeilen) und 9 (3000 Zeilen) statt 2 und 2" in t4 and "8 bis 9 statt 4 bis 6 Mengen" in t4
