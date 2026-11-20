"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio)."""

import random
from dataclasses import dataclass

import streamlit as st

import erv_constants as C


def _bool(text):
    if str(text).lower() in ("true", "1", "ja"):
        return True
    if str(text).lower() in ("false", "0", "nein"):
        return False
    raise ValueError(text)


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None      # bei Auswahlreglern die erlaubten Werte; `alt` ist ein anderer gültiger Wert (für Tests und die Hervorhebung)

    @property
    def alt(self):
        if self.options is not None:
            return next(o for o in self.options if o != self.default)
        return self.hi if self.default != self.hi else self.lo


SETTING_SPECS = {
    "klasse_select": SettingSpec("klasse", str, C.DEFAULT_KLASSE, options=C.KLASSEN),
    "n_radio": SettingSpec("n", int, C.DEFAULT_N, options=C.N_TRAIN),
    "menge_select": SettingSpec("menge", int, C.DEFAULT_MENGE, 0, C.MENGE_ALLE),
    "ebene_radio": SettingSpec("ebene", str, C.EBENEN[0], options=C.EBENEN),
    "zeile_slider": SettingSpec("zeile", int, C.DEFAULT_ZEILE, C.ZEILE_MIN, C.ZEILE_MAX),
    "neu_slider": SettingSpec("neu", int, C.DEFAULT_NEU, C.NEU_MIN, C.NEU_MAX),
    "erk_menge_select": SettingSpec("emenge", int, C.DEFAULT_ERK_MENGE, 0, C.N_TAB_MENGEN - 1),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "gepaart_toggle": SettingSpec("gepaart", _bool, False, options=(False, True)),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {
    "klasse": "klasse_select", "n": "n_radio", "menge": "menge_select", "ebene": "ebene_radio", "zeile": "zeile_slider", "neu": "neu_slider",
    "erk_menge": "erk_menge_select", "m": "m_slider", "gepaart": "gepaart_toggle", "k": "k_slider", "seed": "seed_input",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf erlaubte Werte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if spec.options is not None:
                if value not in spec.options:
                    continue
            else:
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                if state_key == "m_slider":
                    value = int(round(value / C.M_STEP) * C.M_STEP)
                if state_key == "k_slider":
                    value = int(round(value / C.K_STEP) * C.K_STEP)
            st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name].get(key, SETTING_SPECS[state_key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State.
    Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS, dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
