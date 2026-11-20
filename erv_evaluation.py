"""Auswertung und Verdikte (reine Funktionen: `verdict_*` liefern (Art, Text), die App ruft `getattr(st, Art)(Text)`).

Alles liest die eingefrorenen Daten (`erv_tables.py`); nur Experiment 2 würfelt live (Tabellenabruf mit `erv_sampler`, SplitMix64).
Zähleinheiten (im Text immer genannt): Paare von Neu-Trainings (45 aus 10), Paare von Läufen des Erklärers, Zeilenmengen (zehn zu je 20 Zeilen), Zellen aus Klasse, Trainingsgröße und Zeilenmenge (80)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import erv_constants as C
import erv_rank as R
import erv_sampler as S
import erv_schichten as SC
import erv_tables as T

ALLE = C.MENGE_ALLE


@dataclass(frozen=True)
class Settings:
    klasse: str = C.DEFAULT_KLASSE
    n: int = C.DEFAULT_N
    menge: int = C.DEFAULT_MENGE            # 0 bis 9 = Zeilenmenge 1 bis 10, 10 = alle 200 Zeilen
    zeile: int = C.DEFAULT_ZEILE            # 1 bis 20 innerhalb der Zeilenmenge (bei "alle": Zeilenmenge 1)
    ebene: str = C.EBENEN[0]
    neu: int = C.DEFAULT_NEU
    m: int = C.DEFAULT_M
    gepaart: bool = False
    k: int = C.DEFAULT_K
    erk_menge: int = C.DEFAULT_ERK_MENGE    # 0 bis 2: die Zeilenmengen mit eingefrorener Wertetabelle
    seed: int = C.DEFAULT_SEED


# ------------------------------------------------------------------------------------------------------------------------------ Hilfen


def fmt(x, nd=2):
    """Zahl mit Dezimalkomma."""
    return f"{x:.{nd}f}".replace(".", ",")


def pct(x, nd=0):
    return f"{100 * x:.{nd}f} %".replace(".", ",")


def menge_text(menge):
    return "alle 200 Zeilen" if menge == ALLE else f"Zeilenmenge {menge + 1}"


def zeilenindex(s):
    """Index (0 bis 199) der gewählten Zeile; bei „alle 200“ zählt die Zeile in Zeilenmenge 1."""
    return (0 if s.menge == ALLE else s.menge) * C.N_ZEILEN + s.zeile - 1


def sel_menge(menge):
    return None if menge == ALLE else menge


# ------------------------------------------------------------------------------------------------------------------------------ Experiment 1


@lru_cache(maxsize=None)
def daten_alle_mengen(conf, neu):
    """Schicht Daten je Zeilenmenge (10) und alle 200 (11 Zeilen): Mittel über die Paare der ersten `neu` Neu-Trainings. Rückgabe (11, 4): Rangkorrelation Beitrag, gleiches Rang 1,
    Rangkorrelation Wichtigkeit, gleiches Rang 1 der Wichtigkeit. Zeilenmenge ist die Zähleinheit; die Spanne über die zehn Mengen steht in `spanne`."""
    b = T.load()
    return np.array([SC.mittel(R.paar_masse(b.phi[conf, :neu, SC.set_slice(sel_menge(m))])) for m in range(C.N_MENGEN + 1)])


def spanne(werte):
    """(Mittel, kleinster, größter Wert) über die zehn Zeilenmengen (nan ausgelassen)."""
    w = np.asarray(werte, dtype=float)[:C.N_MENGEN]
    return float(np.nanmean(w)), float(np.nanmin(w)), float(np.nanmax(w))


@dataclass(frozen=True)
class NeuTraining:
    balken: np.ndarray            # (neu, 6) Beitrag der Zeile bzw. Wichtigkeit, je Neu-Training
    wahrheit: np.ndarray          # (6,) Wahrheit je Funktional (Hintergrund wie Neu-Training 1)
    paare: np.ndarray             # Rangkorrelationen je Paar von Neu-Trainings (Beitrag der gewählten Zeile bzw. Wichtigkeit der Menge)
    gleich1: float                # Anteil der Paare mit gleichem Rang-1-Merkmal
    menge_beitrag: float          # Mittel über die Paare, Beitrag (Mittel über die Zeilen der Menge)
    menge_wicht: float
    beitrag_spanne: tuple         # (Mittel, kleinster, größter) über zehn Zeilenmengen
    wicht_spanne: tuple
    je_menge_beitrag: np.ndarray  # (10,) für das Band
    je_menge_wicht: np.ndarray
    n_paare: int


def neu_training(s: Settings) -> NeuTraining:
    b = T.load()
    conf = T.config(s.klasse, s.n)
    ni = C.N_TRAIN.index(s.n)
    sl = SC.set_slice(sel_menge(s.menge))
    phi = b.phi[conf, :s.neu]
    a, c = R.pair_indices(s.neu)
    if s.ebene == C.EBENEN[0]:
        r = zeilenindex(s)
        balken = phi[:, r, :]
        truth = b.truth[ni, 0, r, :]
        paare = R.spearman(np.abs(balken[a]), np.abs(balken[c]))
        gleich1 = float((R.rang1(np.abs(balken[a])) == R.rang1(np.abs(balken[c]))).mean())
    else:
        balken = R.wichtigkeit(phi[:, sl])
        truth = R.wichtigkeit(b.truth[ni, 0, sl])
        paare = R.spearman(balken[a], balken[c])
        gleich1 = float((R.rang1(balken[a]) == R.rang1(balken[c])).mean())
    alle = daten_alle_mengen(conf, s.neu)
    return NeuTraining(balken, truth, paare, gleich1, float(alle[sel_idx(s.menge), 0]), float(alle[sel_idx(s.menge), 2]), spanne(alle[:, 0]), spanne(alle[:, 2]),
                       alle[:C.N_MENGEN, 0], alle[:C.N_MENGEN, 2], len(a))


def sel_idx(menge):
    """Zeile der Tabelle `daten_alle_mengen`: 0 bis 9 = Zeilenmenge, 10 = alle 200."""
    return int(menge)


def verdict_neutraining(klasse_name, n, beitrag, wicht, n_paare):
    """beitrag, wicht: (Mittel, kleinster, größter Wert) über zehn Zeilenmengen. Art: warning unter 0,7, info unter 0,9, sonst success (Rangkorrelation des Beitrags)."""
    kind = "warning" if beitrag[0] < 0.7 else ("info" if beitrag[0] < 0.9 else "success")
    return kind, (
        f"{klasse_name}, {n} Trainingszeilen: Rangkorrelation der Beiträge zweier Neu-Trainings im Mittel **{fmt(beitrag[0])}** "
        f"({fmt(beitrag[1])} bis {fmt(beitrag[2])} über zehn Zeilenmengen zu 20 Zeilen; {n_paare} Paare je Menge), der Wichtigkeit **{fmt(wicht[0])}** "
        f"({fmt(wicht[1])} bis {fmt(wicht[2])}). Wichtigkeit ist das Mittel der Beträge über die Zeilen einer Menge."
    )


# ------------------------------------------------------------------------------------------------------------------------------ Experiment 2


@dataclass(frozen=True)
class ErklaererLauf:
    laeufe: np.ndarray        # (k, 6)
    exakt: np.ndarray         # (6,)
    l2: float                 # Median des relativen L2-Abstands zur Aufzählung
    anders1: float            # Anteil der Läufe mit anderem Rang-1-Merkmal als die Aufzählung
    rang_laeufe: float        # Rangkorrelation zwischen Läufen (Mittel über Paare), nan, wenn undefiniert
    kosten: int


def erklaerer_lauf(s: Settings) -> ErklaererLauf:
    """Live: k Läufe des Stichproben-Erklärers für die gewählte Zeile aus der eingefrorenen Wertetabelle (Modell: Historie 0, Modell-Zufalls-Seed 0)."""
    b = T.load()
    conf = T.config(s.klasse, s.n)
    tab = b.tab[conf, s.erk_menge * C.N_ZEILEN + s.zeile - 1]
    from erv_exact import shapley_from_values
    exakt = shapley_from_values(tab.mean(axis=-1))
    m = s.m                                  # der Regler steht in Zehnerschritten, M ist gerade: gepaart sind es M / 2 Reihenfolgen und ihre Umkehrungen
    laeufe = S.run_many(tab, m, s.k, s.seed, paired=s.gepaart)
    l2 = float(np.median(R.rel_l2(laeufe, exakt[None])))
    anders = float((R.rang1(np.abs(laeufe)) != R.rang1(np.abs(exakt))).mean())
    a, c = R.pair_indices(s.k)
    rho = R.spearman(np.abs(laeufe[a]), np.abs(laeufe[c]))
    rl = float(R.nanmean(rho))
    return ErklaererLauf(laeufe, exakt, l2, anders, rl, S.cost(m))


def verdict_erklaerer(m, kosten, gesamt, l2, anders, n_laeufe):
    """warning, sobald ein Lauf ein anderes Rang-1-Merkmal nennt als die Aufzählung (auch bei kleinem Abstand), sonst success."""
    text = (f"Mit **{m}** Reihenfolgen ({kosten} von {gesamt} Auswertungen je Zeile) liegt der relative L2-Abstand dieser Zeile im Median bei **{fmt(l2, 3)}**; "
            f"in **{pct(anders)}** der {n_laeufe} Läufe nennt der Erklärer ein anderes Rang-1-Merkmal als die Aufzählung.")
    if anders > 0:
        return "warning", text + " Ein kleiner Abstand schließt ein anderes Rang-1-Merkmal nicht aus (Rangmaß und Betragsmaß sagen Verschiedenes)."
    return "success", text + " In dieser Zeile stimmt das Rang-1-Merkmal in allen Läufen mit der Aufzählung überein (andere Zeilen können abweichen)."


def erk_tabelle(conf):
    """Eingefrorene Schicht Erklärer (Mittel über die zehn Zeilenmengen; 4 Neu-Trainings x 10 Läufe je Menge, alle 20 Zeilen) für die Zelle: je Variante (Name, M, Mittel, kleinster, größter)
    von relativem L2, Anteil Rang 1 wie die Aufzählung und Rangkorrelation zwischen Läufen."""
    b = T.load()
    zellen = b.erk[conf]
    zeilen = []
    for var, name in (("u", "ungepaart"), ("p", "gepaart (Umkehrung, gleiche Hintergrundzeile)")):
        for m in SC.M_LISTE:
            key = f"{var}{m}"
            if key not in zellen[0]:
                continue
            row = {"Variante": name, "Reihenfolgen M": m, "Auswertungen je Zeile": S.cost(m)}
            for feld, titel in (("l2", "relativer L2"), ("r1", "Rang 1 wie Aufzählung"), ("rk", "Rangkorrelation zwischen Läufen")):
                w = [z[key][feld] for z in zellen]
                row[titel] = (float(np.mean(w)), float(np.min(w)), float(np.max(w)))
            zeilen.append(row)
    return zeilen


def erk_tabelle_text(conf):
    """`erk_tabelle` als Textzeilen für die Anzeige: „Mittel (kleinster bis größter)“ mit Dezimalkomma."""
    def band(w, nd, prozent=False):
        f = (lambda v: pct(v)) if prozent else (lambda v: fmt(v, nd))
        return f"{f(w[0])} ({f(w[1])} bis {f(w[2])})"
    return [{"Variante": r["Variante"], "M": r["Reihenfolgen M"], "Auswertungen je Zeile": r["Auswertungen je Zeile"], "Relativer L2": band(r["relativer L2"], 3),
             "Rang 1 wie Aufzählung": band(r["Rang 1 wie Aufzählung"], 0, True), "Rangkorrelation zwischen Läufen": band(r["Rangkorrelation zwischen Läufen"], 2)} for r in erk_tabelle(conf)]


def paarung_zaehlung(b=None):
    """Zählung über die 80 Zellen (Boosting und Netz x zwei Größen x M = 30 und 100 x zehn Zeilenmengen): in wie vielen hat die gepaarte Fassung den größeren relativen L2-Abstand,
    in wie vielen den kleineren Anteil Rang 1 wie die Aufzählung. Rückgabe (n_zellen, k_l2, k_r1)."""
    b = b or T.load()
    n = kl2 = kr1 = 0
    for conf in range(C.N_CONFIG):
        if "p30" not in b.erk[conf][0]:
            continue
        for z in b.erk[conf]:
            for m in SC.M_LISTE:
                n += 1
                kl2 += z[f"p{m}"]["l2"] > z[f"u{m}"]["l2"]
                kr1 += z[f"p{m}"]["r1"] < z[f"u{m}"]["r1"]
    return n, int(kl2), int(kr1)


def verdict_paarung(n, k_l2, k_r1):
    return "info", (f"Ehrlicher Negativbefund: Die Paarung (Reihenfolge und ihre Umkehrung mit derselben Hintergrundzeile) hilft dem Stichproben-Erklärer in dieser Bauart (M/2 Reihenfolgen und ihre Umkehrung) hier nicht: "
                    f"in **{k_l2} von {n}** Zellen ist der relative L2-Abstand größer als ungepaart, in **{k_r1} von {n}** der Anteil Rang 1 wie die Aufzählung kleiner "
                    f"(Boosting und Netz, zwei Trainingsgrößen, M = 30 und 100, zehn Zeilenmengen). Das gilt für diesen Permutations-Schätzer, nicht für gepaartes KernelSHAP.")


# ------------------------------------------------------------------------------------------------------------------------------ Experiment 3


@lru_cache(maxsize=2)
def alle_zellen(b):
    """Schichten für alle 80 Zellen (8 Zellen x 10 Zeilenmengen): Rangkorrelation des Beitrags und der Wichtigkeit je Schicht.
    Rückgabe ein Feld (8, 10, 3, 2): Schicht (Daten, Modell, Erklärer mit M = 100), Ebene (Beitrag, Wichtigkeit); dazu (8, 10, 2) für den Erklärer mit M = 30. b: Bündel der Daten."""
    out = np.zeros((C.N_CONFIG, C.N_MENGEN, 3, 2))
    out30 = np.zeros((C.N_CONFIG, C.N_MENGEN, 2))
    for conf in range(C.N_CONFIG):
        for m in range(C.N_MENGEN):
            d = SC.mittel(SC.daten_schicht(b.phi[conf], m))
            mo = SC.mittel(SC.modell_schicht(b.phi[conf], m))
            z = b.erk[conf][m]
            out[conf, m] = [[d[0], d[2]], [mo[0], mo[2]], [z["u100"]["rk"], z["u100"]["rkw"]]]
            out30[conf, m] = [z["u30"]["rk"], z["u30"]["rkw"]]
    return out, out30


def schichten_zaehlung(b=None):
    """Zählung über alle 80 Zellen: in wie vielen ist die Rangkorrelation beim Neu-Training die kleinste der drei Schichten (Erklärer mit M = 100), und kleiner als beim Erklärer mit M = 30.
    Rückgabe ein Wörterbuch mit n und den Zählungen für Beitrag und Wichtigkeit."""
    a, a30 = alle_zellen(b or T.load())
    erg = {"n": C.N_CONFIG * C.N_MENGEN}
    for e, name in enumerate(("beitrag", "wichtigkeit")):
        d, mo, er = a[:, :, 0, e], a[:, :, 1, e], a[:, :, 2, e]
        erg[f"{name}_kleinste_100"] = int((d < np.minimum(mo, er)).sum())
        erg[f"{name}_kleiner_als_30"] = int((d < a30[:, :, e]).sum())
        erg[f"{name}_kleiner_als_modell"] = int((d < mo).sum())
        erg[f"{name}_kleiner_als_erklaerer_100"] = int((d < er).sum())
    return erg


def schichten_werte(s: Settings):
    """Die drei Schichten für die gewählte Zelle und Zeilenmenge (bei „alle 200“: Mittel über die zehn Mengen); (3, 2) für Beitrag und Wichtigkeit."""
    a, _ = alle_zellen(T.load())
    conf = T.config(s.klasse, s.n)
    return a[conf].mean(axis=0) if s.menge == ALLE else a[conf, s.menge]


def verdict_schichten(z):
    """z: Ausgabe von `schichten_zaehlung`."""
    n = z["n"]
    return "info", (f"Über alle **{n}** Zellen (4 Modellklassen x 2 Trainingsgrößen x 10 Zeilenmengen) ist das Neu-Training beim **Beitrag** in **{z['beitrag_kleinste_100']}** Zellen die Schicht "
                    f"mit der kleinsten Rangkorrelation (gegen den Modell-Zufalls-Seed und den Erklärer mit 100 Reihenfolgen), gegen den Erklärer mit 30 Reihenfolgen in "
                    f"**{z['beitrag_kleiner_als_30']}**; bei der **Wichtigkeit** in **{z['wichtigkeit_kleinste_100']}** bzw. **{z['wichtigkeit_kleiner_als_30']}** von {n} (bei den gewählten Einstellungen: M = 100 bzw. 30).")


# ------------------------------------------------------------------------------------------------------------------------------ Experiment 4


@dataclass(frozen=True)
class Stabil:
    art: np.ndarray               # (neu,) Anteil Art an (Art + Vorlauf) je Neu-Training auf der gewählten Menge
    wahrheit_mengen: np.ndarray   # (10,) Anteil der Wahrheit je Funktional je Hintergrund (Neu-Training) auf der gewählten Menge
    spitze: np.ndarray            # (neu,) Rang-1-Merkmal der Wichtigkeit je Neu-Training
    k_einheitlich: int            # in wie vielen der zehn Zeilenmengen nennen alle gewählten Neu-Trainings dasselbe Spitzenmerkmal
    art_spanne: tuple             # Mittel (über Neu-Trainings), kleinster, größter Wert über zehn Zeilenmengen
    wahrheit_spanne: tuple        # Mittel (über zehn Hintergründe), kleinster, größter Wert der Wahrheit je Funktional über die zehn Zeilenmengen
    wahrheit_art_vorn: int        # Zeilenmengen, in denen die Wahrheit je Funktional bei allen 10 Hintergründen Art vorn hat
    wicht_gegen_wahrheit: float
    spitze_name: str


def stabil(s: Settings) -> Stabil:
    b = T.load()
    conf = T.config(s.klasse, s.n)
    ni = C.N_TRAIN.index(s.n)
    phi = b.phi[conf, :s.neu]
    sl = SC.set_slice(sel_menge(s.menge))
    art = R.art_anteil(R.wichtigkeit(phi[:, sl]))
    wmen = R.art_anteil(R.wichtigkeit(b.truth[ni][:, sl]))
    spitze = R.rang1(R.wichtigkeit(phi[:, sl]))
    je_menge = np.array([R.art_anteil(R.wichtigkeit(phi[:, SC.set_slice(m)])).mean() for m in range(C.N_MENGEN)])
    einh = sum(bool((R.rang1(R.wichtigkeit(phi[:, SC.set_slice(m)])) == R.rang1(R.wichtigkeit(phi[:, SC.set_slice(m)]))[0]).all()) for m in range(C.N_MENGEN))
    wje = np.array([R.art_anteil(R.wichtigkeit(b.truth[ni][:, SC.set_slice(m)])).mean() for m in range(C.N_MENGEN)])
    vorn = sum(bool((R.rang1(R.wichtigkeit(b.truth[ni][:, SC.set_slice(m)])) == 0).all()) for m in range(C.N_MENGEN))
    haeufig = int(np.bincount(spitze, minlength=6).argmax())
    wgw = float(R.nanmean(R.spearman(R.wichtigkeit(phi[:, sl]), R.wichtigkeit(b.truth[ni][:s.neu, sl]))))
    return Stabil(art, wmen, spitze, einh, (float(je_menge.mean()), float(je_menge.min()), float(je_menge.max())),
                  (float(wje.mean()), float(wje.min()), float(wje.max())), vorn, wgw, C.MERKMALE[haeufig])


def verdict_menge(menge_name, beitrag_menge, wicht_menge, beitrag_spanne, ebene_wicht):
    """Zeilenmenge als Zähleinheit: dieselben Modelle, andere Menge, anderer Wert. beitrag_spanne: (Mittel, kleinster, größter) über zehn Zeilenmengen."""
    lo, hi = beitrag_spanne[1], beitrag_spanne[2]
    return "info", (f"In **{menge_name}** liegt die Rangkorrelation der Beiträge zweier Neu-Trainings bei **{fmt(beitrag_menge)}**, die der Wichtigkeit bei **{fmt(wicht_menge)}**. "
                    f"Über die zehn Zeilenmengen reicht die Rangkorrelation der Beiträge von {fmt(lo)} bis {fmt(hi)}: eine Zahl ohne Zeilenmenge wäre beliebig."
                    + (" (Die Balken zeigen die Wichtigkeit dieser Menge.)" if ebene_wicht else ""))


def verdict_wahrheit_rang1(k_art_vorn):
    """k_art_vorn: in wie vielen der zehn Zeilenmengen die Wahrheit je Funktional bei allen zehn Hintergründen die Art vorn hat."""
    return "info", (f"In **{k_art_vorn} von 10** Zeilenmengen hat die Wahrheit je Funktional (je Hintergrund einer Historie) bei allen zehn Hintergründen die Art vorn: "
                    "das Rang-1-Merkmal ist bei diesen zwei Merkmalen keine Kennzahl, der Anteil mit Spanne ist es.")


def verdict_lime(lime_text):
    return "info", lime_text


def verdict_stabil(klasse_name, n, art_spanne, wahrheit, exakt, k_einheitlich, spitze_name, n_neu):
    """art_spanne: (Mittel, kleinster, größter) des Art-Anteils über zehn Zeilenmengen; wahrheit: Mittel der Wahrheit je Funktional derselben Mengen; exakt: exakte Bevölkerung.
    Art: warning, wenn der Art-Anteil im Mittel mehr als 0,1 unter der Wahrheit je Funktional liegt (stabil, aber daneben), sonst info."""
    daneben = art_spanne[0] < wahrheit - 0.1
    text = (f"{klasse_name}, {n} Trainingszeilen: In **{k_einheitlich} von 10** Zeilenmengen nennen alle {n_neu} Neu-Trainings dasselbe Spitzenmerkmal der Wichtigkeit "
            f"(am häufigsten: {spitze_name}); der Anteil der Art an (Art + Vorlauf) liegt im Mittel bei **{fmt(art_spanne[0])}** ({fmt(art_spanne[1])} bis {fmt(art_spanne[2])} je Zeilenmenge) "
            f"gegen {fmt(wahrheit)} der Wahrheit je Funktional (exakte Bevölkerung: {fmt(exakt, 3)}).")
    if daneben:
        return "warning", text + " Stabil heißt hier nicht richtig: die Neu-Trainings stimmen miteinander überein und liegen im Art-Anteil gemeinsam daneben."
    return "info", text + " Hier liegt der Art-Anteil im Mittel nahe an der Wahrheit je Funktional; das Spitzenmerkmal kann trotzdem zwischen Neu-Trainings und Zeilenmengen wechseln."


# ------------------------------------------------------------------------------------------------------------------------------ zweite Messung


def zellen_einheitlich(b):
    """Je Zelle (8): in wie vielen der zehn Zeilenmengen nennen alle zehn Neu-Trainings dasselbe Spitzenmerkmal der Wichtigkeit, und in wie vielen hat die Wahrheit je Funktional bei allen
    zehn Hintergründen die Art vorn. Rückgabe (einheitlich (8,), art_vorn (8,))."""
    einh = np.array([sum(SC.einheitlich(b.phi[c], m) for m in range(C.N_MENGEN)) for c in range(C.N_CONFIG)])
    vorn = np.array([sum(bool((R.rang1(R.wichtigkeit(b.truth[c % 2][:, SC.set_slice(m)])) == 0).all()) for m in range(C.N_MENGEN)) for c in range(C.N_CONFIG)])
    return einh, vorn


def zweit_zahlen():
    """Zahlen der zweiten Messung (andere Historien und Zeilenmengen) neben denen der Erstmessung, für die Hinweise unter Experiment 3 und 4."""
    erst, zweit = T.load(), T.load_zweit()
    z1, z2 = schichten_zaehlung(erst), schichten_zaehlung(zweit)
    e1, v1 = zellen_einheitlich(erst)
    e2, v2 = zellen_einheitlich(zweit)
    return {"schichten_erst": z1, "schichten_zweit": z2, "einheitlich_erst": e1, "einheitlich_zweit": e2, "art_vorn_erst": v1, "art_vorn_zweit": v2}


def text_zweit_schichten(z):
    a, b_ = z["schichten_erst"], z["schichten_zweit"]
    return (f"Zweite Messung mit anderen Historien und Zeilenmengen (Zufalls-Seeds 300 bis 309 und 43001 bis 43010; die Zahlen oben sind die der ersten, 200 bis 209 und 41001 bis 41010): "
            f"Beitrag: Neu-Training kleinste Schicht in {b_['beitrag_kleinste_100']} statt {a['beitrag_kleinste_100']} von {b_['n']} Zellen (gegen den Erklärer mit 30 Reihenfolgen {b_['beitrag_kleiner_als_30']} statt {a['beitrag_kleiner_als_30']}); "
            f"Wichtigkeit: {b_['wichtigkeit_kleinste_100']} statt {a['wichtigkeit_kleinste_100']} (gegen 30 Reihenfolgen {b_['wichtigkeit_kleiner_als_30']} statt {a['wichtigkeit_kleiner_als_30']}).")


def text_zweit_spitze(z):
    e1, e2, v1, v2 = z["einheitlich_erst"], z["einheitlich_zweit"], z["art_vorn_erst"], z["art_vorn_zweit"]
    baum = lambda e: f"{int(min(e[[0, 1, 2, 3, 4, 5]]))} bis {int(max(e[[0, 1, 2, 3, 4, 5]]))}"  # noqa: E731
    return (f"Zweite Messung (andere Historien und Zeilenmengen): Bei den Baumklassen nennen alle zehn Neu-Trainings dasselbe Spitzenmerkmal in {baum(e2)} von 10 Zeilenmengen (erste Messung: {baum(e1)}), "
            f"beim Netz in {int(e2[6])} (300 Zeilen) und {int(e2[7])} (3000 Zeilen) statt {int(e1[6])} und {int(e1[7])}; die Wahrheit je Funktional hat bei allen zehn Hintergründen die Art vorn in "
            f"{int(min(v2))} bis {int(max(v2))} statt {int(min(v1))} bis {int(max(v1))} Mengen. Zählungen dieser Art hängen an den gezogenen Historien und Zeilenmengen.")
