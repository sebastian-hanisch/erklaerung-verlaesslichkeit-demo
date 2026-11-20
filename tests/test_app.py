"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: laedt app.py mit den
Standardeinstellungen, klickt jeden Preset-Button und faehrt jeden Slider an seine Grenzen und prueft, dass
kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere
st.plotly_chart-Aufrufe ohne eindeutiges key= koennen zufaellig identischen Inhalt rendern und kollidieren)
und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max)."""

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "\u25b6"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_jedes_preset_laeuft_und_zeigt_drei_kernfragen_und_vier_experimente():
    import erv_constants as C
    for name in C.PRESETS:
        at = _fresh_app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])
        assert len([m for m in at.markdown if m.value.startswith("**") and m.value[2] in "123"]) == 3
        assert [h.value for h in at.subheader if h.value.startswith("🔬")] == [
            "🔬 Experiment 1: Neu-Training – dieselbe Antwort?", "🔬 Experiment 2: Gleiches Modell, anderer Lauf des Erklärers",
            "🔬 Experiment 3: Woher kommt die Streuung?", "🔬 Experiment 4: Stabil ist nicht richtig"]
        assert any(h.value == "🚧 Wo die Annahmen enden" for h in at.subheader)


def test_jede_auswahl_der_selectboxen_und_radios_laeuft():
    at = _fresh_app()
    for key, optionen in (("klasse_select", ["gbm", "gbm_ts", "rf", "mlp"]), ("n_radio", [300, 3000]), ("menge_select", list(range(11))), ("erk_menge_select", [0, 1, 2]),
                          ("ebene_radio", None)):
        widget = (at.sidebar.selectbox(key=key) if key.endswith("_select") else at.sidebar.radio(key=key))
        for o in (optionen or list(widget.options)):
            widget.set_value(o).run(timeout=TIMEOUT)
            assert not at.exception, (key, o, [str(e) for e in at.exception])
            widget = (at.sidebar.selectbox(key=key) if key.endswith("_select") else at.sidebar.radio(key=key))


def test_toggle_gepaart_und_seed_knopf():
    at = _fresh_app()
    at.sidebar.toggle(key="gepaart_toggle").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception
    alt = at.sidebar.number_input(key="seed_input").value
    at.sidebar.button[-1].click().run(timeout=TIMEOUT)          # 🎲 Neuer Zufalls-Seed
    assert not at.exception and at.sidebar.number_input(key="seed_input").value != alt


def test_permalink_wird_begrenzt_und_uebernommen():
    at = AppTest.from_file(APP_PATH)
    at.query_params["klasse"] = "rf"
    at.query_params["n"] = "3000"
    at.query_params["menge"] = "99"      # außerhalb: auf 10 (alle 200) begrenzt
    at.query_params["m"] = "13"          # auf Zehnerschritte gerundet
    at.query_params["gepaart"] = "true"
    at.query_params["ebene"] = "gibt es nicht"  # ungültiger Wert: Standard bleibt
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.selectbox(key="klasse_select").value == "rf" and at.sidebar.radio(key="n_radio").value == 3000
    assert at.sidebar.selectbox(key="menge_select").value == 10 and at.sidebar.slider(key="m_slider").value == 10
    assert at.sidebar.toggle(key="gepaart_toggle").value is True
    assert at.sidebar.radio(key="ebene_radio").value == "Beitrag einer Zeile"
