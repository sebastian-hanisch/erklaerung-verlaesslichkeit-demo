"""Claims: jede Zahl der README (Tabelle „Zahl → Test“) wird hier aus den eingefrorenen Daten über dieselben Auswertungsfunktionen wie in der App nachgerechnet.

Die Daten (`data/`) sind einmal gerechnet und eingefroren; die Tests lesen sie und zählen über das GANZE Raster (4 Modellklassen x 2 Trainingsgrößen x 10 Zeilenmengen = 80 Zellen),
keine Zahl aus einer einzelnen Zelle. Was hier als Spanne steht, ist (Mittel, kleinster, größter Wert) über zehn Zeilenmengen; Zellen sind Klassen x Größen, Klassenmittel das Mittel über die
zehn Zeilenmengen. Die Neurechnung der Dateien mit scikit-learn prüft `tests/test_frozen.py`, die Wahrheit der Bevölkerung `tests/test_oracle_wahrheit.py`."""

import numpy as np
import pytest

import erv_constants as C
import erv_evaluation as ev
import erv_rank as R
import erv_schichten as SC
import erv_tables as T

BAEUME = (0, 1, 2, 3, 4, 5)       # Zellen 0 bis 5: Boosting, Boosting mit Teilziehung, Wald (je 300 und 3000)
NETZ = (6, 7)
G300, G3000 = (0, 2, 4, 6), (1, 3, 5, 7)


def klassenmittel(conf, spalte):
    return ev.spanne(ev.daten_alle_mengen(conf, 10)[:, spalte])


def bereich(werte, nd=2):
    return round(min(werte), nd), round(max(werte), nd)


def erk_mittel(b, conf, key, feld):
    return float(np.mean([z[key][feld] for z in b.erk[conf]]))


def erk_spanne(b, conf, key, feld):
    w = [z[key][feld] for z in b.erk[conf]]
    return float(np.mean(w)), float(min(w)), float(max(w))


# ------------------------------------------------------------------------------------------------------------------ B1 bis B3: Schicht Daten und Modell


def test_b1_neutraining_beitrag_rangkorrelation():
    assert bereich([klassenmittel(c, 0)[0] for c in G300]) == (0.44, 0.77)
    assert bereich([klassenmittel(c, 0)[0] for c in G3000]) == (0.63, 0.86)
    assert bereich([klassenmittel(c, 0)[1] for c in G300] + [klassenmittel(c, 0)[2] for c in G300]) == (0.37, 0.81)      # je Zeilenmenge, 300 Zeilen
    assert bereich([klassenmittel(c, 0)[1] for c in G3000] + [klassenmittel(c, 0)[2] for c in G3000]) == (0.57, 0.88)
    assert ev.neu_training(ev.Settings()).n_paare == 45                                                                   # 45 Paare aus 10 Neu-Trainings


def test_b1_ein_einzelwert_gegen_die_tabelle():
    """Die Beispielzeile des Verdikts: Random Forest, 300 Trainingszeilen: 0,44 (0,37 bis 0,54), Wichtigkeit 0,74 (0,58 bis 0,88)."""
    s = ev.Settings(klasse="rf", n=300)
    nt = ev.neu_training(s)
    assert tuple(round(x, 2) for x in nt.beitrag_spanne) == (0.44, 0.37, 0.54) and tuple(round(x, 2) for x in nt.wicht_spanne) == (0.74, 0.58, 0.88)
    assert ev.verdict_neutraining("Random Forest", 300, nt.beitrag_spanne, nt.wicht_spanne, nt.n_paare)[0] == "warning"


def test_b2_neutraining_wichtigkeit_rangkorrelation():
    assert bereich([klassenmittel(c, 2)[0] for c in G300]) == (0.74, 0.91)
    assert bereich([klassenmittel(c, 2)[0] for c in G3000]) == (0.91, 0.96)
    assert bereich([klassenmittel(c, 2)[1] for c in G300]) == (0.58, 0.87) and bereich([klassenmittel(c, 2)[2] for c in G300]) == (0.88, 0.98)
    assert bereich([klassenmittel(c, 2)[1] for c in G3000]) == (0.76, 0.93) and bereich([klassenmittel(c, 2)[2] for c in G3000]) == (0.95, 1.0)


def test_b3_modell_zufalls_seed_beitrag_und_wichtigkeit(bundle):
    erw = {0: (0.97, 0.99), 1: (0.99, 1.00), 2: (0.77, 0.92), 3: (0.86, 0.97), 4: (0.84, 0.95), 5: (0.92, 0.97), 6: (0.93, 0.98), 7: (0.94, 0.99)}
    for conf, (beitrag, wicht) in erw.items():
        mo = np.array([SC.mittel(SC.modell_schicht(bundle.phi[conf], m)) for m in range(10)])
        assert round(mo[:, 0].mean(), 2) == beitrag and round(mo[:, 2].mean(), 2) == wicht, conf
    assert len(SC.modell_schicht(bundle.phi[0], 0)) == 15                         # 5 Historien x 3 Paare aus 3 Zufalls-Seeds


# ------------------------------------------------------------------------------------------------------------------ B4 bis B7: Schicht Erklärer


def test_b4_erklaerer_100_reihenfolgen(bundle):
    assert bereich([erk_mittel(bundle, c, "u100", "rk") for c in range(8)]) == (0.79, 0.93)
    assert bereich([erk_mittel(bundle, c, "u100", "l2") for c in range(8)]) == (0.10, 0.15)
    assert bereich([erk_mittel(bundle, c, "u100", "r1") for c in range(8)]) == (0.91, 0.93)
    anders = [1 - z["u100"]["r1"] for c in range(8) for z in bundle.erk[c]]
    assert round(min(anders), 2) == 0.03 and round(max(anders), 2) == 0.15                  # je Zeilenmenge 3 bis 15 %
    assert bereich([1 - erk_mittel(bundle, c, "u100", "r1") for c in range(8)]) == (0.07, 0.09)       # im Klassenmittel 7 bis 9 %
    l2 = [z["u100"]["l2"] for c in range(8) for z in bundle.erk[c]]
    assert round(min(l2), 2) == 0.09 and round(max(l2), 2) == 0.19                          # je Zeilenmenge 0,09 bis 0,19
    assert ev.erk_tabelle(0)[1]["Auswertungen je Zeile"] == 601


def test_b5_erklaerer_30_reihenfolgen(bundle):
    assert bereich([erk_mittel(bundle, c, "u30", "rk") for c in range(8)]) == (0.66, 0.87)
    assert bereich([erk_mittel(bundle, c, "u30", "l2") for c in range(8)]) == (0.19, 0.27)
    assert bereich([erk_mittel(bundle, c, "u30", "r1") for c in range(8)]) == (0.84, 0.88)
    assert bereich([z["u30"]["r1"] for c in range(8) for z in bundle.erk[c]]) == (0.77, 0.93)       # je Zeilenmenge
    assert ev.erk_tabelle(0)[0]["Auswertungen je Zeile"] == 181


def test_b5_wichtigkeit_des_netzes_im_erklaerer_100(bundle):
    """Betragsmaß klein, Spitzenmerkmal trotzdem nicht immer gleich: relativer L2 der Wichtigkeit 0,026 bis 0,027, Rang 1 der Wichtigkeit wie die Aufzählung in 93 % (300) bzw. 97 % (3000)."""
    assert [round(erk_mittel(bundle, c, "u100", "l2w"), 3) for c in NETZ] == [0.027, 0.026]
    assert [round(erk_mittel(bundle, c, "u100", "r1w"), 2) for c in NETZ] == [0.93, 0.97]
    assert round(min(z["u100"]["r1w"] for c in NETZ for z in bundle.erk[c]), 2) == 0.72      # je Zeilenmenge bis 0,72


def test_b6_schichten_zaehlung_ueber_80_zellen():
    z = ev.schichten_zaehlung()
    assert z["n"] == 80 == C.N_CONFIG * C.N_MENGEN
    assert z["beitrag_kleinste_100"] == 80 and z["beitrag_kleiner_als_modell"] == 80 and z["beitrag_kleiner_als_erklaerer_100"] == 80
    assert z["beitrag_kleiner_als_30"] == 78
    assert z["wichtigkeit_kleinste_100"] == 78 and z["wichtigkeit_kleiner_als_30"] == 78 and z["wichtigkeit_kleiner_als_modell"] == 79 and z["wichtigkeit_kleiner_als_erklaerer_100"] == 78


def test_b6_welche_zellen_sind_die_ausnahmen():
    """Die Ausnahmen beim Beitrag gegen den Erklärer mit 30 Reihenfolgen und bei der Wichtigkeit liegen im Netz mit 3000 Zeilen (Zelle 7) und im Netz mit 300 Zeilen."""
    a, a30 = ev.alle_zellen(T.load())
    ausnahmen_beitrag30 = [(c, m) for c in range(8) for m in range(10) if not a[c, m, 0, 0] < a30[c, m, 0]]
    assert len(ausnahmen_beitrag30) == 2 and {c for c, _ in ausnahmen_beitrag30} <= {6, 7}
    ausnahmen_wicht = [(c, m) for c in range(8) for m in range(10) if not a[c, m, 0, 1] < min(a[c, m, 1, 1], a[c, m, 2, 1])]
    assert len(ausnahmen_wicht) == 2 and {c for c, _ in ausnahmen_wicht} <= {6, 7}


def test_b7_paarung_hilft_nicht():
    n, kl2, kr1 = ev.paarung_zaehlung()
    assert (n, kl2, kr1) == (80, 80, 79)                                                    # 2 Klassen x 2 Größen x 2 M x 10 Zeilenmengen
    b = T.load()
    mittel = lambda var, m, feld: np.mean([z[f"{var}{m}"][feld] for c in (0, 1, 6, 7) for z in b.erk[c]])   # noqa: E731
    assert (round(mittel("u", 30, "l2"), 3), round(mittel("p", 30, "l2"), 3)) == (0.211, 0.262)
    assert (round(mittel("u", 100, "l2"), 3), round(mittel("p", 100, "l2"), 3)) == (0.115, 0.143)
    assert (round(mittel("u", 30, "r1"), 3), round(mittel("p", 30, "r1"), 3)) == (0.863, 0.822)
    assert (round(mittel("u", 100, "r1"), 3), round(mittel("p", 100, "r1"), 3)) == (0.923, 0.899)


# ------------------------------------------------------------------------------------------------------------------ B8 bis B11: stabil ist nicht richtig


def test_b8_art_anteil_der_modelle_gegen_die_wahrheit(bundle):
    art = {c: ev.stabil(ev.Settings(klasse=C.KLASSEN[c // 2], n=C.N_TRAIN[c % 2])).art_spanne for c in range(8)}
    assert bereich([art[c][0] for c in (0, 1, 2, 3, 4, 5)]) == (0.31, 0.36)
    assert bereich([art[c][0] for c in NETZ]) == (0.52, 0.53)
    assert round(min(art[c][1] for c in BAEUME), 2) == 0.21 and round(max(art[c][2] for c in BAEUME), 2) == 0.47          # je Zeilenmenge 0,21 bis 0,47
    assert round(min(art[c][1] for c in NETZ), 2) == 0.49 and round(max(art[c][2] for c in NETZ), 2) == 0.59
    for c in range(8):
        sb = ev.stabil(ev.Settings(klasse=C.KLASSEN[c // 2], n=C.N_TRAIN[c % 2]))
        assert round(sb.wahrheit_spanne[0], 2) == 0.54 and round(sb.wahrheit_spanne[1], 2) == 0.52 and round(sb.wahrheit_spanne[2], 2) == 0.59
    assert round(bundle.wahrheit["art_anteil_exakt"], 3) == 0.537


def test_b9_b10_spitzenmerkmal_und_wahrheit_art_vorn(bundle):
    einh, vorn = ev.zellen_einheitlich(bundle)
    assert [int(x) for x in einh] == [7, 9, 7, 9, 7, 8, 2, 2]
    assert [int(x) for x in vorn] == [6, 4, 6, 4, 6, 4, 6, 4]
    for c in BAEUME:                                                                          # das Spitzenmerkmal der Bäume ist in einheitlichen Mengen der Vorlauf, das des Netzes die Art
        for m in range(10):
            if SC.einheitlich(bundle.phi[c], m):
                assert SC.spitzen(bundle.phi[c], m)[0] == 4
    for c in NETZ:
        for m in range(10):
            if SC.einheitlich(bundle.phi[c], m):
                assert SC.spitzen(bundle.phi[c], m)[0] == 0


def test_b9_beispiel_boosting_menge_5_wechselt(bundle):
    """Boosting, 300 Zeilen, Zeilenmenge 5: Spitzenmerkmal je Neu-Training Art 3, Kunde 1, Vorlauf 6 (in 7 von 10 Mengen sind alle Neu-Trainings einig, in dieser nicht)."""
    zaehl = np.bincount(SC.spitzen(bundle.phi[0], 4), minlength=6)
    assert zaehl[0] == 3 and zaehl[1] == 1 and zaehl[4] == 6 and not SC.einheitlich(bundle.phi[0], 4)


def test_b11_wichtigkeit_gegen_wahrheit_rangkorrelation(bundle):
    erw = {0: 0.40, 1: 0.78, 2: 0.60, 3: 0.79, 4: 0.70, 5: 0.87, 6: 0.95, 7: 0.98}
    for c, soll in erw.items():
        w = [SC.wichtigkeit_gegen_wahrheit(bundle.phi[c], bundle.truth[c % 2], m) for m in range(10)]
        assert round(float(np.mean(w)), 2) == soll, c
    w0 = [SC.wichtigkeit_gegen_wahrheit(bundle.phi[0], bundle.truth[0], m) for m in range(10)]
    assert round(min(w0), 2) == 0.05 and round(max(w0), 2) == 0.60                          # Boosting 300: 0,05 bis 0,60 je Zeilenmenge


def test_b12_gleiches_rang1_der_wichtigkeit_je_paar(bundle):
    erw = {6: (0.62, 0.44, 1.0), 7: (0.68, 0.47, 1.0), 0: (0.85, 0.27, 1.0), 5: (0.89, 0.27, 1.0)}
    for c, (mittel, lo, hi) in erw.items():
        a = ev.spanne(ev.daten_alle_mengen(c, 10)[:, 3])
        assert (round(a[0], 2), round(a[1], 2), round(a[2], 2)) == (mittel, lo, hi), c


def test_b13_r2_gegen_mu(bundle):
    assert [round(float(x), 2) for x in bundle.r2] == [0.71, 0.96, 0.76, 0.96, 0.82, 0.91, 0.88, 0.97]


# ------------------------------------------------------------------------------------------------------------------ B14: zweite Messung (andere Historien, andere Zeilenmengen)


def test_b14_zweite_messung_zaehlungen(zweit):
    z = ev.schichten_zaehlung(zweit)
    assert (z["beitrag_kleinste_100"], z["beitrag_kleiner_als_30"], z["wichtigkeit_kleinste_100"], z["wichtigkeit_kleiner_als_30"]) == (80, 77, 74, 71)
    n, kl2, kr1 = ev.paarung_zaehlung(zweit)
    assert (n, kl2, kr1) == (80, 80, 77)
    einh, vorn = ev.zellen_einheitlich(zweit)
    assert [int(x) for x in einh] == [5, 7, 6, 8, 7, 7, 3, 9]
    assert [int(x) for x in vorn] == [9, 8, 9, 8, 9, 8, 9, 8]


def test_b14_zweite_messung_neutraining_und_art_anteil(zweit):
    a300 = [ev.spanne(np.array([SC.mittel(SC.daten_schicht(zweit.phi[c], m)) for m in range(10)])[:, 0])[0] for c in G300]
    a3000 = [ev.spanne(np.array([SC.mittel(SC.daten_schicht(zweit.phi[c], m)) for m in range(10)])[:, 0])[0] for c in G3000]
    assert bereich(a300) == (0.43, 0.77) and bereich(a3000) == (0.59, 0.86)
    art = [float(np.mean([R.art_anteil(R.wichtigkeit(zweit.phi[c, :10, SC.set_slice(m)])).mean() for m in range(10)])) for c in range(8)]
    assert bereich(art[:6]) == (0.35, 0.41) and bereich(art[6:]) == (0.53, 0.54)
    assert [round(float(x), 2) for x in zweit.r2] == [0.72, 0.96, 0.75, 0.96, 0.79, 0.90, 0.85, 0.96]


# ------------------------------------------------------------------------------------------------------------------ Wahrheit der Menge von 20 Zeilen


def test_wahrheit_je_funktional_hintergrund_40_zeilen_liegt_zwischen_049_und_068(bundle):
    """Wahrheit je Funktional (Art-Anteil) je Zeilenmenge und Hintergrund einer Historie: kleinster und größter der 100 Werte je Trainingsgröße."""
    w = np.array([[[R.art_anteil(R.wichtigkeit(bundle.truth[ni][h:h + 1, SC.set_slice(m)])).mean() for h in range(10)] for m in range(10)] for ni in range(2)])
    assert round(float(w.min()), 2) == 0.47 and round(float(w.max()), 2) == 0.68
    assert round(float(w.mean()), 2) == 0.54
