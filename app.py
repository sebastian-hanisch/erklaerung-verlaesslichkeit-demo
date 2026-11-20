"""Wie verlässlich ist ein Beitrag? Neu-Training, Zufall und Zeilenmenge - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Vier Experimente auf einem erfundenen Vehikel (Container-Verweilzeit, bekannte Funktion mu): Neu-Training, Zufall des Erklärers, Schichten des Zufalls, stabil gegen richtig.
Die Modelle sind eingefroren (data/, erzeugt von tools/erv_make_tables.py); nur der Stichproben-Erklärer in Experiment 2 würfelt live (SplitMix64).
Aufbau und Regeln: DEMO-PLAYBOOK.md (§2). Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import erv_constants as C
import erv_evaluation as ev
import erv_sampler as S
import erv_tables as T
import erv_texts as X
from erv_layout import dunkles_design, kopf_css
from erv_presets import init_session_state_defaults, load_permalink_settings, randomize_seed, sidebar_presets, sync_query_params
from erv_visualization import build_art_anteil, build_balken, build_band, build_laeufe, build_paare, build_schichten, build_spitzen

st.set_page_config(page_title=f"{X.TITEL} – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _lauf(settings):
    return ev.erklaerer_lauf(settings)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in erv_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title(f"🔍 {X.TITEL}")
st.markdown(X.EINLEITUNG)

with st.expander("So funktioniert die Messung", expanded=True):
    st.markdown(X.SO_FUNKTIONIERT)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Beispielszenario ist hervorgehoben
    st.header("⚙️ Modell und Zeilen")
    klasse = st.selectbox("Modellklasse", C.KLASSEN, key="klasse_select", format_func=lambda k: C.KLASSEN_NAMEN[k],
                          help="Vier Klassen mit festen Einstellungen. Das Netz ist auf die bekannte Funktion abgestimmt (bewusst das glatteste brauchbare).")
    n = st.radio("Trainingszeilen", C.N_TRAIN, key="n_radio", horizontal=True, help="Größe jeder Historie, auf der ein Modell trainiert wurde.")
    menge = st.selectbox("Zeilenmenge", list(range(C.MENGE_ALLE + 1)), key="menge_select",
                         format_func=lambda m: "alle 200 Zeilen" if m == C.MENGE_ALLE else f"Menge {m + 1} (20 Zeilen)",
                         help="Zehn getrennte Mengen zu je 20 Zeilen oder alle 200 zusammen. Jede Zahl der Experimente gilt für diese Menge; die Spanne über die zehn Mengen steht dabei.")
    ebene = st.radio("Ebene (Experiment 1)", C.EBENEN, key="ebene_radio", help="Beitrag: eine Zeile, mit Vorzeichen. Wichtigkeit: Mittel der Beträge über die Zeilen der Menge.")
    zeile = st.slider("Zeile in der Menge", C.ZEILE_MIN, C.ZEILE_MAX, key="zeile_slider",
                      help="Die erklärte Zeile (Beitrag in Experiment 1 und Zeile des Erklärers in Experiment 2). Bei „alle 200 Zeilen“ zählt sie in Menge 1.")
    neu = st.slider("Zahl der Neu-Trainings", C.NEU_MIN, C.NEU_MAX, key="neu_slider", help="Wie viele der zehn Historien in Experiment 1 und 4 verglichen werden.")
    st.header("🎲 Erklärer (Experiment 2)")
    erk_menge = st.selectbox("Zeilenmenge des Erklärers", list(range(C.N_TAB_MENGEN)), key="erk_menge_select", format_func=lambda m: f"Menge {m + 1}",
                             help="Für die Stichprobe liegen die Wertetabellen der Mengen 1 bis 3 vor.")
    m = st.slider("Reihenfolgen M", C.M_MIN, C.M_MAX, step=C.M_STEP, key="m_slider", help="Wie viele Reihenfolgen der Merkmale der Erklärer zieht. Kosten: M mal 6 plus 1 Auswertungen je Zeile.")
    gepaart = st.toggle("Gepaart (Umkehrung)", key="gepaart_toggle", help="Je Reihenfolge auch die umgekehrte, mit derselben Hintergrundzeile (M/2 Paare).")
    k = st.slider("Läufe K", C.K_MIN, C.K_MAX, step=C.K_STEP, key="k_slider", help="Wie oft der Erklärer unabhängig läuft.")
    seed = st.number_input("Zufalls-Seed des Erklärers", 0, C.SEED_MAX, key="seed_input", step=1)
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

sync_query_params({"klasse_select": klasse, "n_radio": int(n), "menge_select": int(menge), "ebene_radio": ebene, "zeile_slider": int(zeile), "neu_slider": int(neu),
                   "erk_menge_select": int(erk_menge), "m_slider": int(m), "gepaart_toggle": bool(gepaart), "k_slider": int(k), "seed_input": int(seed)})

s = ev.Settings(klasse, int(n), int(menge), int(zeile), ebene, int(neu), int(m), bool(gepaart), int(k), int(erk_menge), int(seed))
b = T.load()
conf = T.config(s.klasse, s.n)
kname = C.KLASSEN_KURZ[s.klasse]
nt = ev.neu_training(s)
lauf = _lauf(s)
sb = ev.stabil(s)
beitrag_ebene = s.ebene == C.EBENEN[0]

# --- Kernfragen (live, vor den Experimenten) ------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Bleibt der Beitrag derselbe, wenn man das Modell neu trainiert?**")
kind, text = ev.verdict_neutraining(kname, s.n, nt.beitrag_spanne, nt.wicht_spanne, nt.n_paare)
getattr(st, kind)(text)
st.markdown("**2. Wie viel Streuung bringt der Erklärer selbst (festes Modell, gewählte Zeile)?**")
kind, text = ev.verdict_erklaerer(s.m, lauf.kosten, S.enumeration_cost(), lauf.l2, lauf.anders1, s.k)
getattr(st, kind)(text)
st.markdown("**3. Heißt stabil auch richtig, und hängt die Antwort an der Zeilenmenge?**")
kind, text = ev.verdict_stabil(kname, s.n, sb.art_spanne, sb.wahrheit_spanne[0], b.wahrheit["art_anteil_exakt"], sb.k_einheitlich, sb.spitze_name, s.neu)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Neu-Training – dieselbe Antwort?")
if beitrag_ebene:
    st.caption(f"Beitrag der Zeile {s.zeile} ({T.zeile_text(ev.zeilenindex(s))}); {s.neu} Neu-Trainings, je eine andere Historie aus demselben Generator. "
               f"Raute: Wahrheit je Funktional (bekannte Funktion μ, Hintergrund wie Neu-Training 1).")
else:
    st.caption(f"Wichtigkeit in {ev.menge_text(s.menge)}: Mittel der Beträge der Beiträge über die Zeilen der Menge; {s.neu} Neu-Trainings. Raute: Wahrheit je Funktional.")
c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(build_balken(nt.balken, nt.wahrheit, f"{kname}, {s.n} Trainingszeilen", beitrag_ebene), width="stretch", key="chart_balken")
with c2:
    gueltig = nt.paare[np.isfinite(nt.paare)]
    mean_paare = float(gueltig.mean()) if len(gueltig) else float("nan")
    st.plotly_chart(build_paare(nt.paare, mean_paare, f"Rangkorrelation je Paar ({nt.n_paare} Paare)"), width="stretch", key="chart_paare")
    mc1, mc2 = st.columns(2)
    mc1.metric("Rangkorrelation, Paare im Mittel", ev.fmt(mean_paare) if len(gueltig) else "nicht definiert")
    mc2.metric("Gleiches Rang 1", ev.pct(nt.gleich1))
st.plotly_chart(build_band(nt.je_menge_beitrag, None if s.menge == C.MENGE_ALLE else s.menge, "Rangkorrelation der Beiträge je Zeilenmenge (Mittel über die Paare)", "Rangkorrelation"),
                width="stretch", key="chart_band")
kind, text = ev.verdict_menge(ev.menge_text(s.menge), nt.menge_beitrag, nt.menge_wicht, nt.beitrag_spanne, not beitrag_ebene)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Gleiches Modell, anderer Lauf des Erklärers")
st.caption(f"Zeile {s.zeile} der Menge {s.erk_menge + 1} ({T.zeile_text(s.erk_menge * C.N_ZEILEN + s.zeile - 1)}); Modell: {kname}, {s.n} Trainingszeilen, erstes Neu-Training. "
           f"{s.k} Läufe zu {s.m} Reihenfolgen {'(gepaart)' if s.gepaart else '(ungepaart)'}; Kosten {lauf.kosten} von {S.enumeration_cost()} Auswertungen je Zeile.")
e1, e2 = st.columns([3, 2])
with e1:
    st.plotly_chart(build_laeufe(lauf.laeufe, lauf.exakt), width="stretch", key="chart_laeufe")
with e2:
    st.metric("Relativer L2-Abstand (Median)", ev.fmt(lauf.l2, 3))
    st.metric("Läufe mit anderem Rang 1", ev.pct(lauf.anders1))
    st.metric("Rangkorrelation zwischen Läufen", ev.fmt(lauf.rang_laeufe) if np.isfinite(lauf.rang_laeufe) else "nicht definiert")
st.markdown("**Eingefrorene Zählung über zehn Zeilenmengen** (je Zeilenmenge 4 Neu-Trainings x 10 Läufe x 20 Zeilen; Mittel, kleinster bis größter Wert):")
st.dataframe(pd.DataFrame(ev.erk_tabelle_text(conf)), hide_index=True, width="stretch")
kind, text = ev.verdict_paarung(*ev.paarung_zaehlung())
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Woher kommt die Streuung?")
st.caption(f"{kname}, {s.n} Trainingszeilen, {ev.menge_text(s.menge)}: Rangkorrelation je Schicht (kleiner = mehr Streuung). Daten und Modell: Mittel über Paare von Neu-Trainings bzw. Zufalls-Seeds; "
           f"Erklärer: 100 Reihenfolgen, Mittel über Läufe (eingefrorene Zählung).")
st.plotly_chart(build_schichten(ev.schichten_werte(s)), width="stretch", key="chart_schichten")
kind, text = ev.verdict_schichten(ev.schichten_zaehlung())
getattr(st, kind)(text)
zz = ev.zweit_zahlen()
st.caption(ev.text_zweit_schichten(zz))

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Stabil ist nicht richtig")
st.caption(f"{kname}, {s.n} Trainingszeilen, {s.neu} Neu-Trainings, {ev.menge_text(s.menge)}: Anteil der Art an Art und Vorlauf der Wichtigkeit; "
           f"die Wahrheit je Funktional gegen die exakte Bevölkerung ({ev.fmt(b.wahrheit['art_anteil_exakt'], 3)}).")
f1, f2 = st.columns(2)
with f1:
    st.plotly_chart(build_art_anteil(sb.art, sb.wahrheit_mengen, b.wahrheit["art_anteil_exakt"]), width="stretch", key="chart_art")
with f2:
    st.plotly_chart(build_spitzen(sb.spitze, s.neu), width="stretch", key="chart_spitzen")
kind, text = ev.verdict_wahrheit_rang1(sb.wahrheit_art_vorn)
getattr(st, kind)(text)
st.caption(ev.text_zweit_spitze(zz))
kind, text = ev.verdict_lime(X.LIME_ZEILE)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(X.GRENZEN)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(X.FORMELN_LESEHILFE)
    st.markdown(X.FORMELN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
