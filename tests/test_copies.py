"""Kopie-Test: Generator und Zufallsstrom sind Kopien (nicht Importe) und werden gegen eingefrorene Werte der Quellen gepinnt.

Quelle des Generators: verweilzeit-demo/vwz_generator.py (Funktionen kunden_effekte, log_median_and_s, draw_features, make_history), die Werte unten stammen aus dem Lauf der Quelle
(2026-11-19). Quelle des Zufallsstroms: sebastianhanisch-website/tools/templates/rng_splitmix64_exact.py (Quelltext-Hash, nicht ast.dump: der hängt an der Python-Version)."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import erv_generator as G
from erv_rng import SplitMix64

ROOT = Path(__file__).resolve().parent.parent
RNG_SHA256 = "d69b2b2fd67c1234cc0d79249676a662a2189d02803eea7d2edfb04b7a505a3f"


def test_rng_ist_wortgleiche_kopie_der_vorlage():
    text = (ROOT / "erv_rng.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(text).hexdigest() == RNG_SHA256


def test_rng_feste_werte_der_vorlage():
    r = SplitMix64(12345)
    assert [r.next_u64() for _ in range(3)] == [2454886589211414944, 3778200017661327597, 2205171434679333405]
    assert r.random() == pytest.approx(0.17611780724496118, abs=1e-15)
    assert r.normal() == pytest.approx(-0.6061905461687908, abs=1e-12)
    assert r.below(40) == 26


@pytest.mark.parametrize("n, seed, summe_x, summe_y, y0", [
    (50, 200, 467.28265459770347, 67.16108197672816, -0.4672692599049397),
    (3000, 209, 27747.858850274257, 4218.729088583282, 0.1011966544863275),
    (20, 41001, 160.44415119824038, 24.706146849918216, -0.5647734015280171),
])
def test_generator_feste_werte_der_quelle(n, seed, summe_x, summe_y, y0):
    X, y = G.daten(n, seed)
    assert X.shape == (n, 6)
    assert X.sum() == pytest.approx(summe_x, rel=1e-9)
    assert y.sum() == pytest.approx(summe_y, rel=1e-9)
    assert y[0] == pytest.approx(y0, rel=1e-9)


def test_generator_erste_zeile_und_wahrheit():
    X, _ = G.daten(50, 200)
    assert X[0].tolist() == pytest.approx([2.0, 9.0, 0.0, 0.0, 0.7222548961406137, -0.2781541667594208])
    mu, s = G.log_median_and_s(np.array([0, 1, 2, 3]), np.array([0, 5, 11, 3]), np.array([1, 0, 0, 0]), np.array([1, 0, 1, 0]), np.array([0.0, 4.0, 2.5, 0.0]))
    assert mu.tolist() == pytest.approx([1.8256154575589787, 1.3196147424321858, 0.920073844452777, 1.8681581785749939])
    assert s.tolist() == [0.55, 0.25, 0.35, 0.9]
    assert G.UK[0] == pytest.approx(-0.42714751093638936) and G.UK[11] == pytest.approx(0.2905490717255771)
