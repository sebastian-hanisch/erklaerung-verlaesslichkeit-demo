"""Feste Texte der App (einmal hier, damit README und App denselben Wortlaut haben; `tests/test_texts.py` vergleicht)."""

TITEL = "Wie verlässlich ist ein Beitrag? Neu-Training, Zufall und Zeilenmenge"

EINLEITUNG = """
Ein **Beitrag** sagt, wie ein Merkmal die Prognose **einer Zeile** verschiebt; die **Wichtigkeit** ist das Mittel der Beträge über viele Zeilen. Beides gehört nicht „den Daten“, sondern **einem trainierten Modell, einem Hintergrund, einer Zeilenmenge und einem Lauf des Erklärers**. Diese Demo misst das auf einem **erfundenen** Vehikel (Container-Verweilzeit, die wahre Funktion μ ist bekannt) an drei Schichten des Zufalls: **Neu-Training** mit anderer Historie, **anderer Zufalls-Seed des Modells** und **anderer Lauf des Erklärers** bei festem Modell. Sie zeigt außerdem, dass **stabil nicht richtig heißt**.

Weiter unten: vier Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""

SO_FUNKTIONIERT = """
1. **Vehikel:** ein erfundener Generator für Container-Verweilzeiten mit sechs Merkmalen (Art, Kunde, Zoll, Wochenende, Vorlauf, Gewicht als reines Rauschen); Zielgröße ist ln(Verweilzeit). Es gibt keine reale Datenquelle.
2. **Modelle:** vier Klassen mit festen Einstellungen (Boosting, Boosting mit Teilziehung, Random Forest, kleines Netz), trainiert auf 300 oder 3000 Zeilen. Je Zelle aus Klasse und Größe stehen 20 Modelle bereit: zehn Neu-Trainings mit anderer Historie und zusätzliche Zufalls-Seeds des Modells. Alles ist **eingefroren**, die App trainiert nie.
3. **Beitrag:** der interventionale Shapley-Wert mit 40 Trainingszeilen als Hintergrund, **exakt** aus 64 Koalitionen (64 mal 40 = 2560 Modellauswertungen je Zeile). Die Wahrheit je Funktional ist derselbe Beitrag für die bekannte Funktion μ.
4. **Zeilenmenge:** zehn getrennte Mengen zu je 20 Zeilen. Jede Aussage nennt ihre Zeilenmenge und steht als **Mittel und Spanne über zehn Mengen**, nie als Punktzahl aus einer festen Ziehung.
5. **Zwei Maße:** die **Rangkorrelation** (Reihenfolge der Beträge der Beiträge) und der **relative L2-Abstand** (Größe der Beiträge). Das zweite sieht Umordnungen an der Spitze kaum; deshalb stehen beide da.
"""

GRENZEN = """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Ein Vehikel, sechs Merkmale, ein Funktional (interventional)** | Andere Funktionale (konditional, pfadabhängig, Leave-one-out) geben andere Beiträge; hier nicht gemessen. | eigene Demo der Reihe |
| **Feste Modelleinstellungen; das Netz ist auf μ abgestimmt** | Das Netz ist bewusst das glatteste brauchbare; ein schwächer regularisiertes Netz ist nicht gemessen. | - |
| **Zeilenmengen zu 20 Zeilen, Hintergrund 40 Zeilen** | Mengen mit mehr Zeilen, andere Hintergrundgrößen und mehr Merkmale sind nicht gemessen; die Spanne über zehn Mengen ist selbst eine Stichprobe. | - |
| **Der Erklärer ist ein eigener Permutations-Schätzer** | KernelSHAP als Erklärer ist nicht gemessen; die Paarung hilft diesem Schätzer nicht, das gilt nicht für gepaartes KernelSHAP. | eigene Demo der Reihe |
| **Ein festes Modell je Experiment 2** | Die Stichproben des Erklärers laufen auf einem Modell je Zelle (erstes Neu-Training), die Neu-Trainings-Aussagen auf zehn. | - |
| **Kein Intervall, keine Konfidenzaussage** | Die Streuung steht da; ob ein Intervall sie abdeckt, wird hier nicht gemessen. | eigene Demo der Reihe |
| **Gleich gute Modelle und der Wert eines Trainingspunkts** | Die Frage „welches Modell“ ist hier fest beantwortet (eines je Neu-Training). | eigene Demo der Reihe |
| **Kein Nutzen im Betrieb** | Dass Erklärungen in Logistik, Lager oder Fertigung zu besseren Entscheidungen führen, ist nicht gemessen und wird nicht behauptet. | - |
"""

# Formelblock: ein vollständiges Modell der Messgrößen (nur % als \\%, KaTeX liest % als Kommentar)
FORMELN_LESEHILFE = r"""
**Lesehilfe.** Gegeben sind ein trainiertes Modell $f$, eine zu erklärende Zeile $x$ und ein Hintergrund $B$ (40 Trainingszeilen des Modells). Gesucht sind der Beitrag $\phi_j(x)$ jedes Merkmals $j$ und das Maß, wie stark er sich bei Neu-Training, anderem Zufalls-Seed des Modells und anderem Lauf des Erklärers ändert.

| Symbol | Bedeutung |
|---|---|
| $d = 6$ | Zahl der Merkmale |
| $f$ | trainiertes Modell (Boosting, Random Forest, Netz) |
| $x$, $b$ | zu erklärende Zeile, Hintergrundzeile |
| $x_S, b_{\bar S}$ | Zeile mit den Merkmalen aus $S$ von $x$ und allen übrigen von $b$ |
| $v_x(S)$ | Wert der Koalition $S$ (Mittel über den Hintergrund) |
| $\phi_j(x)$ | Beitrag des Merkmals $j$ (Shapley-Wert, exakt) |
| $\hat\phi_j(x)$ | Beitrag aus $M$ gezogenen Reihenfolgen (Stichproben-Erklärer) |
| $I_j$ | Wichtigkeit: Mittel von $\lvert \phi_j \rvert$ über die Zeilen einer Zeilenmenge $Z$ |
| $\rho$ | Rangkorrelation (Spearman, mittlere Ränge bei Gleichstand) |
| $A$ | Anteil der Art an (Art + Vorlauf) der Wichtigkeit |
"""

FORMELN = r"""
**Wertfunktional (interventional) und Beitrag.** Mit dem Hintergrund $B$ ist der Wert einer Koalition $S \subseteq \{1, \dots, d\}$

$$v_x(S) = \frac{1}{|B|} \sum_{b \in B} f\big(x_S,\, b_{\bar S}\big),$$

und der Beitrag des Merkmals $j$ ist der Shapley-Wert (Koalitionsformel, $2^6 = 64$ Koalitionen, $64 \cdot 40 = 2560$ Modellauswertungen je Zeile):

$$\phi_j(x) = \sum_{S \subseteq \{1,\dots,d\} \setminus \{j\}} \frac{|S|!\,(d - |S| - 1)!}{d!}\,\big(v_x(S \cup \{j\}) - v_x(S)\big).$$

Die Summe der Beiträge ist $f(x)$ minus das Mittel von $f$ über $B$ (Effizienz). Die Wahrheit je Funktional ist derselbe Beitrag mit $\mu$ statt $f$.

**Stichproben-Erklärer** (Štrumbelj und Kononenko 2010, Permutationsform). Für $m = 1, \dots, M$ werden eine Reihenfolge $\pi_m$ der Merkmale und eine Hintergrundzeile $b^{(m)}$ gezogen; $S_m^j$ sind die Merkmale vor $j$ in $\pi_m$:

$$\hat\phi_j(x) = \frac{1}{M} \sum_{m=1}^{M} \Big[ f\big(x_{S_m^j \cup \{j\}},\, b^{(m)}_{\overline{S_m^j \cup \{j\}}}\big) - f\big(x_{S_m^j},\, b^{(m)}_{\overline{S_m^j}}\big) \Big].$$

Er ist erwartungstreu. Kosten: $6M + 1$ Auswertungen je Zeile; die Aufzählung ist ab $M = 427$ billiger.

**Wichtigkeit, Anteil, Maße.** Für eine Zeilenmenge $Z$ ist $I_j = \frac{1}{|Z|} \sum_{x \in Z} |\phi_j(x)|$ und $A = I_{\text{Art}} / (I_{\text{Art}} + I_{\text{Vorlauf}})$. Mit den (mittleren) Rängen $r_j$ und $r'_j$ zweier Beitragsvektoren ist

$$\rho = \frac{\sum_j (r_j - \bar r)(r'_j - \bar r')}{\sqrt{\sum_j (r_j - \bar r)^2 \,\sum_j (r'_j - \bar r')^2}}, \qquad \frac{\lVert \hat\phi - \phi \rVert_2}{\lVert \phi \rVert_2}$$

(Rangmaß und Betragsmaß; $\rho$ ist undefiniert, wenn ein Vektor konstant ist, und wird dann gezählt, nicht als 0 gerechnet). Gleiches Rang-1-Merkmal heißt $\arg\max_j |\phi_j|$ stimmt überein; bei Gleichstand gilt der kleinste Index der Reihenfolge Art, Kunde, Zoll, Wochenende, Vorlauf, Gewicht.

Der Formelblock beschreibt die Messgrößen vollständig; das Modell $f$ ist ein beliebiges trainiertes Modell und kein Teil der Formeln.

**Literatur.** Štrumbelj, Kononenko (2010), *An Efficient Explanation of Individual Classifications using Game Theory*, JMLR 11, 1 bis 18 (gelesen). Hwang u. a. (2026), „Explanation Multiplicity“, NeurIPS-2026-Poster (Posterseite gelesen, Volltext nicht). Covert, Lee (2021), AISTATS (nur Zusammenfassung). Krishna u. a. (2024), TMLR (nur Zusammenfassung). Rahnama u. a. (2021) (nur Zusammenfassung).
"""

LIME_ZEILE = (
    "Eine Vormessung der Planung mit einem **eigenen LIME-artigen Surrogat** (nicht das Paket `lime`; eine Zeilenmenge von 60 Zeilen; in dieser Demo **nicht nachgerechnet**) fand bei 5000 Stichproben "
    "eine Rangkorrelation von 0,98 zwischen den Stichproben-Zufalls-Seeds (fast perfekt stabil), aber nur 0,4 bis 0,5 Übereinstimmung mit dem Shapley-Wert bei den Baummodellen (Netz 0,78): "
    "Stabilität über Läufe schließt eine große Abweichung vom Shapley-Wert nicht aus."
)
