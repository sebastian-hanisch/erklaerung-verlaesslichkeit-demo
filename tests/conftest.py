"""Gemeinsame Fixtures: Importpfad, die eingefrorenen Daten und eine kleine Wertetabelle, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import erv_tables as T  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    return T.load()


@pytest.fixture(scope="session")
def zweit():
    return T.load_zweit()


@pytest.fixture
def spiel3():
    """Das Handspiel des Plans (Teil 1) als Wertetabelle über sechs Merkmale: nur die Merkmale 0, 1, 2 zählen, 3 bis 5 sind Nullspieler; ein Hintergrundzeile.
    v(leer) = 0, v(A) = 3, v(B) = 3, v(C) = 2, v(AB) = 4, v(AC) = 4, v(BC) = 3, v(ABC) = 6; Shapley-Wert (5/2, 2, 3/2, 0, 0, 0)."""
    wert = {0: 0, 1: 3, 2: 3, 4: 2, 3: 4, 5: 4, 6: 3, 7: 6}      # Bitmaske der Merkmale 0 bis 2 -> v
    tab = np.array([[wert[k & 7]] for k in range(64)], dtype=float)  # (64, 1)
    return tab
