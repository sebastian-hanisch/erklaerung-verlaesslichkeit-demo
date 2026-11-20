"""Konstanten: Modellklassen, Seeds der eingefrorenen Messung, Regler-Grenzen, Farben und Presets."""

# --- Modellklassen und Trainingsgrößen (feste Einstellungen; sklearn nur in tools/erv_make_tables.py) -----------------------------------------
KLASSEN = ("gbm", "gbm_ts", "rf", "mlp")
KLASSEN_NAMEN = {
    "gbm": "Boosting (200 Bäume, Tiefe 3)",
    "gbm_ts": "Boosting mit Teilziehung (Zeilen und Spalten 0,7)",
    "rf": "Random Forest (60 Bäume)",
    "mlp": "Kleines Netz (16 Einheiten)",
}
KLASSEN_KURZ = {"gbm": "Boosting", "gbm_ts": "Boosting mit Teilziehung", "rf": "Random Forest", "mlp": "Netz"}
N_TRAIN = (300, 3000)
N_CONFIG = len(KLASSEN) * len(N_TRAIN)        # 8 Zellen aus Klasse und Trainingsgröße; Index = Klasse * 2 + Größe
N_HINTERGRUND = 40                            # Hintergrund: die ersten 40 Trainingszeilen des jeweiligen Modells
N_MENGEN = 10                                 # zehn getrennte Zeilenmengen zu je 20 Zeilen
N_ZEILEN = 20
N_ERK_HIST = 4                                # Historien (Neu-Trainings), über die die Schicht Erklärer gemittelt ist
N_LAEUFE_FEST = 10                            # Läufe des Erklärers je Zelle in der eingefrorenen Messung
N_TAB_MENGEN = 3                              # Zeilenmengen mit eingefrorener Wertetabelle (Experiment 2)
MERKMALE = ("Art", "Kunde", "Zoll", "Wochenende", "Vorlauf", "Gewicht")
ART_NAMEN = ("Import", "Export", "Umschlag", "Leer")

# Zwei getrennte Zufalls-Seed-Sätze: die Erstmessung (Daten der App) und die Wiederholung mit anderen Historien und Zeilenmengen
SEEDSETS = {
    "erst": {"hist0": 200, "zeilen0": 41001, "ref": 42001, "erk": 710},
    "zweit": {"hist0": 300, "zeilen0": 43001, "ref": 44001, "erk": 920},
}
MODELL_SEEDS = (0, 1, 2)
N_REF_ZEILEN = 300                            # frische Zeilen für R² gegen mu

# --- Regler ----------------------------------------------------------------------------------------------------------------------------------
EBENEN = ("Beitrag einer Zeile", "Wichtigkeit (Mittel über die Zeilen)")
NEU_MIN, NEU_MAX, DEFAULT_NEU = 2, 10, 10
M_MIN, M_MAX, DEFAULT_M, M_STEP = 10, 400, 100, 10
K_MIN, K_MAX, DEFAULT_K, K_STEP = 10, 100, 30, 10
ZEILE_MIN, ZEILE_MAX, DEFAULT_ZEILE = 1, 20, 5
MENGE_ALLE = 10                               # Wert des Reglers "Zeilenmenge" für alle 200 Zeilen (sonst 0 bis 9)
SEED_MAX, DEFAULT_SEED = 999999, 17
DEFAULT_KLASSE, DEFAULT_N, DEFAULT_MENGE, DEFAULT_ERK_MENGE = "gbm", 300, 0, 1

# --- Darstellung -----------------------------------------------------------------------------------------------------------------------------
COLOR_PALETTE = ("#2F6B65", "#c9742b", "#4c78a8", "#a05195", "#7a8b3a", "#b04a4a", "#5b7083", "#d1a900", "#6b4c9a", "#2d9cdb")
COLOR_TRUTH = "#222222"
COLOR_EXACT = "#222222"
COLOR_LINE = "#b04a4a"

# --- Presets (nur abweichende Regler stehen im Preset; die übrigen gelten mit ihrem Standard) -------------------------------------------------------
PRESETS = {
    "Boosting, 300 Zeilen": {},
    "Random Forest, 300 Zeilen": {"klasse": "rf"},
    "Netz, 3000 Zeilen": {"klasse": "mlp", "n": 3000},
    "Wenige Reihenfolgen (30)": {"m": 30},
    "Andere Zeilenmenge": {"menge": 4},
}
PRESET_HELP = {
    "Boosting, 300 Zeilen": "Voreinstellung: Boosting, 300 Trainingszeilen, Zeilenmenge 1 (20 Zeilen), Beitrag der Zeile 5, zehn Neu-Trainings; der Erklärer zieht 100 Reihenfolgen auf Zeile 5 der Menge 2.",
    "Random Forest, 300 Zeilen": "Wie die Voreinstellung, aber ein Random Forest: dieselben Daten, eine andere Modellklasse.",
    "Netz, 3000 Zeilen": "Ein kleines Netz mit 3000 Trainingszeilen: trifft den Art-Anteil der Wahrheit (das Netz ist auf μ abgestimmt); ob alle Neu-Trainings dasselbe Spitzenmerkmal nennen, hängt an den Historien (Erstmessung 2, zweite Messung 9 von 10 Mengen).",
    "Wenige Reihenfolgen (30)": "Der Erklärer zieht nur 30 statt 100 Reihenfolgen (181 statt 601 von 2560 Auswertungen je Zeile).",
    "Andere Zeilenmenge": "Zeilenmenge 5 statt 1: dieselben Modelle, andere 20 Zeilen; die Antwort ändert sich.",
}
