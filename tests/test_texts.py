"""Texte: README und App sagen dasselbe (zwei Fassungen werden verglichen, nicht von Hand synchron gehalten); keine ASCII-Umlaute, keine verbotenen Wörter in den festen Texten."""

import re
from pathlib import Path

import erv_constants as C
import erv_texts as X

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_grenzen_tabelle_der_app_steht_wortgleich_in_der_readme():
    assert X.GRENZEN.strip() in README


def test_readme_hat_die_pflichtabschnitte_in_der_reihenfolge():
    ueberschriften = re.findall(r"^## (.+)$", README, flags=re.M)
    erwartet = ["Warum dieses Problem", "Modell", "Methodik", "Befunde (gemessen, keine Behauptungen)", "Befunde und Korrekturen", "Ehrliche Grenzen", "Verifikation", "Dateistruktur",
                "🚧 Bewusst nicht enthalten / nicht gemessen", "Quellen", "Lokal ausführen", "Tests ausführen"]
    assert ueberschriften == erwartet


def test_feste_texte_ohne_verbotene_woerter_und_mit_umlauten():
    alle = " ".join([X.EINLEITUNG, X.SO_FUNKTIONIERT, X.GRENZEN, X.FORMELN_LESEHILFE, X.FORMELN, X.LIME_ZEILE, " ".join(C.PRESET_HELP.values()), " ".join(C.PRESETS)])
    for verboten in ("Saatwert", "Schwanz", "trägt", "tragen", "Standardfall", "Phase 0", "geplant", "Stück"):
        assert verboten not in alle, verboten
    for ascii_umlaut in (" ae ", "ue ", "Waehr", "fuer ", "ueber", "koennen"):
        assert ascii_umlaut not in alle.replace("Reihenfolge", ""), ascii_umlaut


def test_keine_scheinpruefungen_in_den_tests():
    for p in (ROOT / "tests").glob("test_*.py"):
        text = p.read_text(encoding="utf-8")
        if p.name == "test_texts.py":
            continue
        assert "or True" not in text and "assert True" not in text, p.name


def test_formeln_haben_kein_ungeschuetztes_prozent():
    """KaTeX liest % als Kommentar (auch in \text{...}): in den Formeltexten steht kein einzelnes Prozentzeichen."""
    for text in (X.FORMELN, X.FORMELN_LESEHILFE):
        assert "%" not in text.replace(r"\%", "")


def test_dezimalkomma_in_den_festen_texten():
    for text in (X.EINLEITUNG, X.SO_FUNKTIONIERT, X.GRENZEN, X.LIME_ZEILE, " ".join(C.PRESET_HELP.values())):
        assert not re.search(r"(?<![\w.])\d\.\d", text), text[:80]
