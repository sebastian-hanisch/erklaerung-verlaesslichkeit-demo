"""Plotly-Abbildungen (alle Achsen fest: fixedrange, damit Touch-Scrollen nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go

import erv_constants as C


def _fest(fig, hoehe=360, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=hoehe, margin=dict(l=50, r=20, t=40, b=50), legend=dict(orientation="h", y=-0.25), **layout)
    return fig


def build_balken(balken, wahrheit, ebene_text, vorzeichen):
    """Je Neu-Training ein Balken je Merkmal (Beitrag mit Vorzeichen oder Wichtigkeit), die Wahrheit je Funktional als Raute."""
    fig = go.Figure()
    for i, zeile in enumerate(balken):
        fig.add_bar(x=list(C.MERKMALE), y=zeile, name=f"Neu-Training {i + 1}", marker_color=C.COLOR_PALETTE[i % len(C.COLOR_PALETTE)], legendgroup="neu", showlegend=len(balken) <= 6 or i == 0)
    fig.add_scatter(x=list(C.MERKMALE), y=wahrheit, mode="markers", name="Wahrheit je Funktional (bekannte Funktion μ)",
                    marker=dict(symbol="diamond", size=11, color=C.COLOR_TRUTH, line=dict(width=1, color="white")))
    fig.update_layout(barmode="group", title=ebene_text, yaxis_title="Beitrag (ln-Verweilzeit)" if vorzeichen else "mittleres |Beitrag| (ln-Verweilzeit)")
    return _fest(fig)


def build_paare(paare, mittel, titel):
    """Rangkorrelation je Paar von Neu-Trainings als Streifen; senkrechte Linie = Mittel."""
    fig = go.Figure()
    p = np.asarray(paare, dtype=float)
    p = p[np.isfinite(p)]
    rng = np.random.default_rng(1)
    fig.add_scatter(x=p, y=rng.uniform(-0.3, 0.3, len(p)), mode="markers", name="Paar", marker=dict(color=C.COLOR_PALETTE[2], size=7, opacity=0.7))
    fig.add_vline(x=mittel, line_color=C.COLOR_LINE, line_dash="dash", annotation_text=f"Mittel {mittel:.2f}".replace(".", ","))
    fig.update_xaxes(range=[-1.05, 1.05], title="Rangkorrelation (Spearman)")
    fig.update_yaxes(visible=False, range=[-1, 1])
    fig.update_layout(title=titel, showlegend=False)
    return _fest(fig, hoehe=240)


def build_band(je_menge, gewaehlt, titel, ytitel):
    """Ein Wert je Zeilenmenge (1 bis 10), die gewählte hervorgehoben, gestrichelt das Mittel über die zehn Mengen."""
    x = list(range(1, len(je_menge) + 1))
    fig = go.Figure()
    farben = [C.COLOR_LINE if (gewaehlt is not None and i == gewaehlt) else C.COLOR_PALETTE[2] for i in range(len(je_menge))]
    fig.add_bar(x=x, y=je_menge, marker_color=farben, name=ytitel)
    fig.add_hline(y=float(np.mean(je_menge)), line_dash="dash", line_color=C.COLOR_TRUTH)
    fig.update_xaxes(title="Zeilenmenge (je 20 Zeilen)", dtick=1)
    fig.update_yaxes(title=ytitel, range=[0, 1.05])
    fig.update_layout(title=titel, showlegend=False)
    return _fest(fig, hoehe=300)


def build_laeufe(laeufe, exakt):
    """Punktwolke der Läufe je Merkmal, die Aufzählung als Raute."""
    fig = go.Figure()
    rng = np.random.default_rng(2)
    for j, name in enumerate(C.MERKMALE):
        fig.add_scatter(x=[name] * len(laeufe), y=laeufe[:, j], mode="markers", name=name, showlegend=False,
                        marker=dict(color=C.COLOR_PALETTE[j], size=6, opacity=0.55), hoverinfo="y")
    fig.add_scatter(x=list(C.MERKMALE), y=exakt, mode="markers", name="Aufzählung (exakt)", marker=dict(symbol="diamond", size=12, color=C.COLOR_EXACT, line=dict(width=1, color="white")))
    fig.update_layout(title="Läufe des Erklärers (Punkte) gegen die Aufzählung (Raute)", yaxis_title="Beitrag (ln-Verweilzeit)", xaxis=dict(categoryorder="array", categoryarray=list(C.MERKMALE)))
    _ = rng
    return _fest(fig)


def build_schichten(werte):
    """werte (3, 2): Schicht (Daten, Modell, Erklärer) x Ebene (Beitrag, Wichtigkeit): Rangkorrelation."""
    fig = go.Figure()
    namen = ["Daten (Neu-Training)", "Modell-Zufalls-Seed", "Erklärer (100 Reihenfolgen)"]
    for e, (name, farbe) in enumerate((("Beitrag einer Zeile", C.COLOR_PALETTE[0]), ("Wichtigkeit", C.COLOR_PALETTE[1]))):
        fig.add_bar(x=namen, y=werte[:, e], name=name, marker_color=farbe, text=[f"{v:.2f}".replace(".", ",") for v in werte[:, e]], textposition="outside")
    fig.update_yaxes(range=[0, 1.1], title="Rangkorrelation (Spearman)")
    fig.update_layout(barmode="group", title="Drei Schichten des Zufalls (Mittel über die Paare)")
    return _fest(fig, hoehe=330)


def build_art_anteil(art, wahrheit_mengen, exakt):
    """Art-Anteil je Neu-Training (Punkte) gegen die Wahrheit je Funktional je Hintergrund (Band) und die exakte Bevölkerung (Linie)."""
    fig = go.Figure()
    fig.add_hrect(y0=float(np.min(wahrheit_mengen)), y1=float(np.max(wahrheit_mengen)), fillcolor=C.COLOR_TRUTH, opacity=0.10, line_width=0,
                  annotation_text="Wahrheit je Funktional (zehn Hintergründe)", annotation_position="top left")
    fig.add_hline(y=exakt, line_dash="dash", line_color=C.COLOR_TRUTH, annotation_text=f"exakte Bevölkerung {exakt:.3f}".replace(".", ","), annotation_position="bottom right")
    fig.add_scatter(x=[f"Neu-Training {i + 1}" for i in range(len(art))], y=art, mode="markers", marker=dict(size=11, color=C.COLOR_PALETTE[0]), name="Art-Anteil im Modell")
    fig.update_yaxes(range=[0, 1], title="Art / (Art + Vorlauf)")
    fig.update_layout(title="Anteil der Art an Art und Vorlauf je Neu-Training", showlegend=False)
    return _fest(fig, hoehe=330)


def build_spitzen(spitze, neu):
    """Anzahl der Neu-Trainings je Spitzenmerkmal der Wichtigkeit."""
    z = np.bincount(spitze, minlength=len(C.MERKMALE))
    fig = go.Figure(go.Bar(x=list(C.MERKMALE), y=z, marker_color=C.COLOR_PALETTE[3]))
    fig.update_yaxes(range=[0, max(neu, 1)], title="Neu-Trainings", dtick=1)
    fig.update_layout(title="Spitzenmerkmal der Wichtigkeit je Neu-Training", showlegend=False)
    return _fest(fig, hoehe=300)
