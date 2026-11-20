# Wie verlässlich ist ein Beitrag? Neu-Training, Zufall und Zeilenmenge – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-erklaerung-verlaesslichkeit-demo.streamlit.app/)** -->

Demo der Reihe **Beitrag und Beleg** (Feld „Modelle verstehen“) in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Die Vorgänger-Demos zeigen *eine* Erklärung *eines* Modells (ein Beitrag, ein Wertfunktional, eine Wahrheit je Funktional). Diese Demo hebt die stille Annahme auf, **diese eine Erklärung gelte für die Daten**: Sie gilt für ein trainiertes Modell, mit einem Hintergrund, auf einer Zeilenmenge, bei einem Lauf des Erklärers. Das ist **maschinelles Lernen als eigenständiges Thema** (Erklärbarkeit), mit einer Brücke zur Optimierung: Prognosen wie die Verweilzeit eines Containers speisen Planungsregeln, und wer eine Erklärung dafür heranzieht, muss wissen, wie stark sie vom Neu-Training und vom Zufall abhängt.

Vehikel ist ein **erfundener** Generator für Container-Verweilzeiten (kopiert aus `verweilzeit-demo`, wahre Mitte μ bekannt, sechs Merkmale: Art, Kunde, Zoll, Wochenende, Vorlauf und Gewicht als reines Rauschen). **Es gibt keine reale Datenquelle.** In den gelesenen Quellen gab es keine Messung, ob Erklärungen von Prognosemodellen in Logistik, Lager oder Fertigung zu besseren Entscheidungen führen; was die Zahlen für einen Betrieb heißen, wird hier nicht behauptet.

## Warum dieses Problem

Eine Erklärung sieht aus wie eine Eigenschaft der Daten („der Vorlauf zählt am meisten“). Sie ist aber die Eigenschaft eines trainierten Modells, und ein Modell hängt an den Trainingsdaten, am Zufall seines Trainings und, wenn der Beitrag geschätzt wird, am Zufall des Erklärers. **Kernfragen:** (1) Bleibt der Beitrag derselbe, wenn man das Modell neu trainiert? (2) Wie viel Streuung bringt der Erklärer selbst? (3) Heißt stabil auch richtig, und hängt die Antwort an der Zeilenmenge?

## Modell

* **Beitrag** (je Zeile) und **Wichtigkeit** (Mittel von |Beitrag| über die Zeilen einer Menge) werden getrennt gezeigt. Wertfunktional ist durchgehend **interventional** (Hintergrund: die ersten 40 Trainingszeilen des Modells); der Shapley-Wert wird **exakt** über 64 Koalitionen aufgezählt (64 mal 40 = 2560 Modellauswertungen je Zeile). Es gibt kein `shap`-Paket: Aufzählung und Stichprobe sind eigener Code mit Orakel.
* **Wahrheit je Funktional**: derselbe Beitrag für die bekannte Funktion μ. „Exakte Bevölkerung“ heißt die analytische Rechnung aus dem Generator (Art-Anteil 0,537, hier gerechnet); „Referenzmenge“ eine große Stichprobe (4 000 Instanzen, Art-Anteil 0,540, Messung der Planung, hier nicht neu gerechnet).
* **Zeilenmenge als Zähleinheit.** Zehn getrennte Mengen zu je 20 Zeilen (Generator-Zufalls-Seeds 41001 bis 41010) und alle 200 zusammen. Jede Aussage nennt ihre Zeilenmenge und steht als **Mittel und Spanne über die zehn Mengen**, nie als Punktzahl aus einer festen Ziehung.
* **Drei Schichten des Zufalls.** *Daten:* zehn Neu-Trainings aus zehn Historien (Daten-Zufalls-Seeds 200 bis 209), Zähleinheit 45 Paare je Zeilenmenge. *Modell:* gleiche Historie, anderer Modell-Zufalls-Seed (Historien 0 bis 4 mit je drei Zufalls-Seeds, 15 Paare). *Erklärer:* gleiches Modell, gleiche Zeilen, Beitrag per Stichprobe (Štrumbelj/Kononenko 2010, Permutationsform: je Stichprobe eine Reihenfolge und eine Hintergrundzeile), 10 Läufe je Zelle, 45 Paare von Läufen. Kosten in Auswertungen je Zeile: 6M + 1 für M Reihenfolgen; 181, 601 und 1801 für M = 30, 100, 300; die Aufzählung (2560) ist ab M = 427 billiger.
* **Zwei Maße, beide gezeigt.** *Rangmaß:* Spearman-Rangkorrelation mit mittleren Rängen bei Gleichstand (undefiniert, wenn eine Seite konstant ist: dann gezählt, nicht als 0 gerechnet) und „gleiches Rang-1-Merkmal“. *Betragsmaß:* relativer L2-Abstand ‖φ̂ − φ‖ / ‖φ‖ (Median). Hwang u. a. 2026 sagen, das L2-Maß sehe Umordnungen der Spitzenmerkmale kaum; hier ist das nachgemessen, nicht übernommen. Bei Gleichstand an der Spitze gilt der kleinste Index der Reihenfolge Art, Kunde, Zoll, Wochenende, Vorlauf, Gewicht.
* **Modellklassen** (immer mindestens zwei, Hyperparameter fest): Boosting (200 Bäume, Tiefe 3, Lernrate 0,1, ohne Teilziehung, hat nur einen kleinen Modell-Zufall durch Gleichstände bei Splits, Rangkorrelation zwischen Modell-Zufalls-Seeds 0,97 bis 1,00), Boosting mit Teilziehung (Zeilen- und Spaltenanteil 0,7, hat Modell-Zufall), Random Forest (60 Bäume, mindestens 3 Zeilen je Blatt, Spaltenanteil 0,5), kleines Netz (eine versteckte Schicht mit 16 Einheiten, L2-Strafe 3, L-BFGS; **auf R² gegen μ abgestimmt**, also bewusst das glatteste brauchbare Netz), je mit 300 oder 3000 Trainingszeilen. R² gegen μ auf 300 frischen Zeilen (Mittel über die zehn Neu-Trainings): 0,71 / 0,76 / 0,82 / 0,88 bei 300 Zeilen und 0,96 / 0,96 / 0,91 / 0,97 bei 3000 (Reihenfolge: Boosting, Boosting mit Teilziehung, Wald, Netz).
* **Eingefroren.** Die App trainiert nie ein Modell (kein scikit-learn zur Laufzeit). `tools/erv_make_tables.py` trainiert 160 Modelle (8 Zellen aus Klasse und Größe mal 20 Modelle) und rechnet je Modell und Zeile die Aufzählung; die Stichprobenläufe von Experiment 2 lesen aus eingefrorenen Wertetabellen (`data/erv_tables.npz`) und würfeln live (SplitMix64; die Vorlage ist so gebaut, dass Windows und Linux dieselben Bits liefern, der Linux-Lauf, Docker, Python 3.12, frische Installation, ist am 2026-11-20 gemacht und grün). Herkunft und Stand: `data/QUELLE.json`.

## Methodik

Vier Experimente, Seitenleiste mit fünf Beispielszenarien (Boosting, 300 Zeilen; Random Forest, 300 Zeilen; Netz, 3000 Zeilen; wenige Reihenfolgen (30); andere Zeilenmenge), drei Kernfragen mit live berechneten Verdikten vor den Experimenten:

1. **Neu-Training: dieselbe Antwort?** Balken je Neu-Training (Beitrag einer Zeile oder Wichtigkeit einer Menge) gegen die Wahrheit je Funktional, Rangkorrelation je Paar, die Spanne über die zehn Zeilenmengen als Band.
2. **Gleiches Modell, anderer Lauf des Erklärers.** Live: K Läufe zu M Reihenfolgen auf einer Zeile, ungepaart oder gepaart (Umkehrung mit derselben Hintergrundzeile); relativer L2-Abstand, Anteil der Läufe mit anderem Rang 1, Rangkorrelation zwischen Läufen. Daneben die eingefrorene Zählung über zehn Zeilenmengen (4 Neu-Trainings mal 10 Läufe mal 20 Zeilen).
3. **Woher kommt die Streuung?** Die drei Schichten nebeneinander, mit der Zählung über alle 80 Zellen.
4. **Stabil ist nicht richtig.** Anteil Art / (Art + Vorlauf) je Neu-Training gegen die Wahrheit je Funktional und die exakte Bevölkerung; Spitzenmerkmal je Neu-Training; Zählung „in k von 10 Zeilenmengen nennen alle Neu-Trainings dasselbe Spitzenmerkmal“.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen gelten für **dieses Vehikel, diese vier Modellklassen mit festen Einstellungen, 300 oder 3000 Trainingszeilen, Zeilenmengen zu 20 Zeilen, Hintergrund 40 Zeilen, interventionales Funktional** und die Historien mit den Zufalls-Seeds 200 bis 209 (die „erste Messung“). „Spanne“ ist, wo nichts anderes steht, kleinster bis größter Wert über zehn Zeilenmengen. Eine zweite Messung mit anderen Historien und Zeilenmengen steht unten.

| Frage | Befund | Test |
|---|---|---|
| Neu-Training: Rangkorrelation der Beträge der Beiträge einer Zeile (Klassenmittel über zehn Mengen, 45 Paare je Menge) | 0,44 bis 0,77 bei 300 und 0,63 bis 0,86 bei 3000 Trainingszeilen; je Zeilenmenge 0,37 bis 0,81 bzw. 0,57 bis 0,88; Beispiel Random Forest, 300 Zeilen: 0,44 (0,37 bis 0,54) | `test_b1_*` (exakt aus den Daten) |
| Neu-Training: Rangkorrelation der Wichtigkeit | 0,74 bis 0,91 (je Menge 0,58 bis 0,98) bei 300 und 0,91 bis 0,96 (0,76 bis 1,00) bei 3000; Random Forest, 300 Zeilen: 0,74 (0,58 bis 0,88) | `test_b2_*` |
| Anderer Zufalls-Seed des Modells, gleiche Historie (Beitrag / Wichtigkeit, 15 Paare) | Boosting 0,97 / 0,99 (300) und 0,99 / 1,00 (3000); mit Teilziehung 0,77 / 0,92 und 0,86 / 0,97; Wald 0,84 / 0,95 und 0,92 / 0,97; Netz 0,93 / 0,98 und 0,94 / 0,99 | `test_b3_*` |
| Erklärer, M = 100 Reihenfolgen (601 von 2560 Auswertungen je Zeile); Klassenmittel | Rangkorrelation zwischen Läufen 0,79 bis 0,93; relativer L2-Abstand 0,10 bis 0,15 (je Menge 0,09 bis 0,19); Rang 1 wie die Aufzählung in 91 bis 93 % der Zeilen, also **anderes Rang 1 in 7 bis 9 %** (je Menge 3 bis 15 %), obwohl der Abstand klein ist | `test_b4_*` |
| Erklärer, M = 30 (181 Auswertungen) | Rangkorrelation 0,66 bis 0,87; relativer L2 0,19 bis 0,27; Rang 1 wie die Aufzählung 84 bis 88 % (je Menge 77 bis 93 %) | `test_b5_erklaerer_30_*` |
| Betragsmaß übersieht beim Netz (Art und Vorlauf fast gleichauf) die Spitze | Wichtigkeit des Netzes, M = 100: relativer L2 0,026 bis 0,027, aber Rang 1 der Wichtigkeit wie die Aufzählung nur in 93 % (300 Zeilen) bzw. 97 % (3000) der Läufe, je Menge bis 72 % | `test_b5_wichtigkeit_*` |
| Welche Schicht streut am meisten (80 Zellen), bei den gewählten Einstellungen (M = 100 bzw. 30) | Das Neu-Training hat beim **Beitrag in 80 von 80 Zellen** die kleinste Rangkorrelation (gegen Modell-Zufalls-Seed und Erklärer mit 100 Reihenfolgen), gegen den Erklärer mit 30 Reihenfolgen in 78 von 80; bei der **Wichtigkeit** in 78 von 80 (beide Fälle). Ausnahmen liegen beim Netz | `test_b6_*` |
| Paarung in dieser Bauart (M/2 Reihenfolgen und ihre Umkehrung, gleiche Hintergrundzeile) im Erklärer | hilft **nicht**: größerer relativer L2 als ungepaart in **80 von 80** Zellen (M = 30: 0,262 gegen 0,211; M = 100: 0,143 gegen 0,115), kleinerer Anteil Rang 1 in 79 von 80 (0,822 gegen 0,863; 0,899 gegen 0,923); Boosting und Netz, zwei Größen, zehn Mengen | `test_b7_*` |
| Art-Anteil an (Art + Vorlauf) der Wichtigkeit | Bäume 0,31 bis 0,36 im Klassenmittel (je Menge 0,21 bis 0,47), Netz 0,52 bis 0,53 (0,49 bis 0,59), Wahrheit je Funktional 0,54 (je Menge 0,52 bis 0,59), exakte Bevölkerung 0,537 | `test_b8_*`, `tests/test_oracle_wahrheit.py` |
| Spitzenmerkmal der Wichtigkeit, alle zehn Neu-Trainings gleich | Bäume in 7 bis 9 von 10 Zeilenmengen (der Vorlauf), Netz in 2 von 10 (die Art); in den übrigen Mengen wechselt es (Beispiel Boosting, 300 Zeilen, Menge 5: Art 3, Kunde 1, Vorlauf 6 von 10 Neu-Trainings) | `test_b9_*` |
| Wahrheit je Funktional hat bei allen zehn Hintergründen die Art vorn | in 6 von 10 Zeilenmengen (300 Zeilen) bzw. 4 von 10 (3000): **Rang 1 ist bei Art und Vorlauf keine Kennzahl**, der Anteil mit Spanne ist es | `test_b9_*` |
| Wichtigkeit gegen Wahrheit je Funktional (Rangkorrelation, Mittel über die Mengen) | Boosting 0,40 (300) und 0,78 (3000); mit Teilziehung 0,60 und 0,79; Wald 0,70 und 0,87; Netz 0,95 und 0,98 | `test_b11_*` |
| Gleiches Rang-1-Merkmal der Wichtigkeit je Paar von Neu-Trainings | Netz 0,62 (300) und 0,68 (3000), Boosting 300 0,85, Wald 3000 0,89; je Menge zwischen 0,27 und 1,00 | `test_b12_*` |
| Handbeispiele von Hand (Brüche, Aufzählung aller Ausgänge) | Erklärer-Zufall bei sechs Reihenfolgen: nur 11933/15552 (76,7 %) richtig, Aufzählen ist exakt; Abstand 0,0202 gegen Spearman 1/2; Zeilenmenge von Hand: Rangkorrelation 1/2, Wurzel 3 / 2, 1 je Menge | `tests/test_handbeispiele.py` |

**Zweite Messung (andere Historien: Zufalls-Seeds 300 bis 309, andere Zeilenmengen: 43001 bis 43010).** Das Neu-Training ist beim Beitrag wieder in 80 von 80 Zellen die kleinste Schicht (gegen 30 Reihenfolgen 77 statt 78); bei der **Wichtigkeit in 74 statt 78** (gegen 30 Reihenfolgen 71). Die Neu-Training-Rangkorrelation des Beitrags liegt im Klassenmittel bei 0,43 bis 0,77 (300) und 0,59 bis 0,86 (3000). **Zählungen zum Spitzenmerkmal hängen an den gezogenen Historien:** Bei den Baumklassen nennen alle zehn Neu-Trainings in 5 bis 8 von 10 Mengen dasselbe Spitzenmerkmal (erste Messung 7 bis 9), beim Netz mit 3000 Zeilen in **9** statt 2; die Wahrheit je Funktional hat bei allen zehn Hintergründen die Art vorn in 8 bis 9 statt 4 bis 6 von 10 Mengen; der Art-Anteil der Bäume liegt im Klassenmittel bei 0,35 bis 0,41 (erste Messung 0,31 bis 0,36), der des Netzes bei 0,53 bis 0,54. Haltbar über beide Messungen ist also: das Neu-Training streut beim Beitrag am meisten, die Bäume liegen im Art-Anteil unter der Wahrheit je Funktional (0,54), die Paarung hilft nicht (80 von 80 beim relativen L2). Nicht haltbar ist jede Zahl der Art „das Netz wechselt das Spitzenmerkmal in 8 von 10 Mengen“. (`test_b14_*`)

**LIME-artiges Surrogat.** Eine Vormessung der Planung (eigener Bau, nicht das Paket `lime`, 60 Zeilen) fand bei 5000 Stichproben Rangkorrelation 0,98 zwischen den Stichproben-Zufalls-Seeds und nur 0,4 bis 0,5 Übereinstimmung mit dem Shapley-Wert bei den Baummodellen (Netz 0,78). **In dieser Demo nicht nachgerechnet**, nur zitiert.

## Befunde und Korrekturen

Abweichungen von Plan und Vormessung. Die Daten-Schicht (Neu-Training, Modell-Zufalls-Seed) stimmt mit der Vormessung bis in die zweite Stelle überein (gleiche Historien, gleiche Modelle); abweichend sind nur Zahlen, in die der Zufall des Erklärers eingeht (hier SplitMix64 statt NumPy-Zufall) oder die Plan und Messung unterschiedlich nennen:

* **Paarung beim Rang-1-Maß:** 79 statt 78 von 80 Zellen; relativer L2 gepaart 0,262 statt 0,264 (M = 30) und 0,143 statt 0,142 (M = 100), ungepaart 0,115 statt 0,116; Rang 1 ungepaart bei M = 30: 0,863 statt 0,858. Die Richtung (80 von 80 beim L2) bleibt.
* **Wichtigkeit des Netzes im Erklärer (M = 100):** Rang 1 wie die Aufzählung in 97 statt 98 % (3000 Zeilen).
* **Ausnahmen der Zählung „kleinste Schicht“:** Beitrag gegen 30 Reihenfolgen 78 von 80 (zwei Ausnahmen im Netz), Wichtigkeit 78 von 80, wie im Plan; die zweite Messung gibt 77 bzw. 74 und 71 (siehe oben).
* **Wahrheit des Art-Anteils:** „exakte Bevölkerung“ ist hier **0,537** (analytisch aus 1224 gewichteten Stützpunkten gerechnet, Quadratur 24 Knoten; mit 12 Knoten 0,538); die 0,540 des Plans sind die Referenzmenge aus 4 000 Instanzen. Die erste Fassung des Plans nannte 0,49 (eine feste Menge); Wahrheit je Funktional bei 20 Zeilen und Hintergrund 40 Zeilen liegt je Menge zwischen 0,52 und 0,59 (Mittel über zehn Hintergründe), je Menge und Hintergrund zwischen 0,47 und 0,68.
* **Spitzenmerkmal „stabil“** gilt nur je Zeilenmenge (7 bis 9 von 10 Mengen) und hängt an der Auswahl der Historien (zweite Messung: 5 bis 8, beim Netz 3000 sogar 9 von 10). Der Satz „stabil, aber falsch“ gilt für den **Art-Anteil** (Bäume liegen in beiden Messungen unter 0,54), nicht für das Spitzenmerkmal.
* **Kein PDF-Export** und keine Wertetabellen für mehr als drei Zeilenmengen (Plan: „drei Zeilenmengen zur Wahl“); Experiment 3 steckt als Zählung auch in Experiment 1 und 2 und bleibt eigener Abschnitt.

## Ehrliche Grenzen

Dieselbe Tabelle wie in der App („🚧 Wo die Annahmen enden“):

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

Ausführlicher:

* Ein Vehikel, sechs Merkmale, **ein** Funktional (interventional); die anderen Funktionale geben andere Beiträge.
* Feste Modelleinstellungen; das Netz ist auf μ abgestimmt (bewusst das glatteste brauchbare), ein schwächer regularisiertes Netz ist nicht gemessen.
* Zeilenmengen zu 20 Zeilen, zehn Mengen, 10 Historien (Modell-Zufalls-Seeds nur für 5 Historien), Hintergrund 40 Zeilen; Mengen mit mehr Zeilen, andere Hintergrundgrößen und mehr als sechs Merkmale sind **nicht gemessen**. Die Spanne über zehn Mengen ist selbst eine Stichprobe (siehe zweite Messung).
* Der Erklärer ist ein **eigener Permutations-Schätzer**, kein KernelSHAP; die Rangunsicherheit von KernelSHAP je Lauf ist nicht gemessen. Die Aussage „Paarung hilft nicht“ gilt für diesen Schätzer, nicht für gepaartes KernelSHAP (dort misst eine Vormessung der Planung einen kleineren Fehler).
* Die Stichproben des Erklärers laufen auf **einem** Modell je Zelle (erstes Neu-Training, Zeilenmengen 1 bis 3); die eingefrorene Zählung über zehn Mengen mittelt vier Neu-Trainings.
* **Plattform:** Zahlen unter Windows, Python 3.14, scikit-learn 1.9. Die eingefrorenen Dateien werden auf jeder Plattform gleich gelesen; die Neurechnung (`tools/erv_make_tables.py --check`) ist im Test ein Band (Boosting 5e-2, Boosting mit Teilziehung und Wald 1e-5, Netz 3e-3). **Linux-Lauf** (Docker, Python 3.12, frische Installation) am 2026-11-20: grün; das Boosting weicht dort bis 0,02 ab (andere Gleichstandsauflösung beim Neu-Training), darum das weitere Band.
* Kein Intervall, keine Konfidenzaussage; **kein Nutzen in Betrieb, Logistik oder Recht belegt**; Menschenstudien (Erklärungen heben Zustimmung, nicht unbedingt Leistung) sind nicht Gegenstand.

## Verifikation

* **Handbeispiele** (Plan, Teil 1 bis 5) als exakte Tests mit Brüchen und mit Aufzählung aller 6^M Ausgänge (`tests/test_handbeispiele.py`); der Schätzer des Repos läuft auf dem Handspiel (sechs Merkmale, drei Nullspieler) und gibt für M = 1 und 2 dieselbe Verteilung.
* **Orakel** (anderer Rechenweg): Stichprobenschätzer mit allen 720 Reihenfolgen mal allen 40 Hintergrundzeilen gleich der Koalitionsformel (Abstand unter 1e-12); Koalitionsformel gleich Mittel über alle n! Reihenfolgen in Brüchen; Rangkorrelation gleich scipy mit und ohne Gleichstände; Effizienz in jedem Lauf; NumPy-Zufallsstrom Bit für Bit gleich der Klasse SplitMix64; Mittel von 3000 Läufen im Band von 4,5 Standardfehlern; Linearfall in geschlossener Form; die Wahrheit der Bevölkerung gegen eine Stichprobe (`tests/test_oracle_erklaerer.py`, `tests/test_oracle_wahrheit.py`).
* **Eigenschaften**, **Randfälle**, **Verdikte** (jede Schwelle, jede Zahl im Text), **Claims** (jede Zahl der Tabelle oben), **eingefrorene Daten** (Form, Wertetabelle gegen Beiträge, Effizienz, Wahrheit gegen den Generator, Neurechnung mit scikit-learn), **Kopie-Test** (Generator und Zufallsstrom gegen feste Werte der Quellen), **Layout** (aktives Beispielszenario, jeder Regler hebt die Markierung auf), **App** (jedes Beispielszenario, jede Auswahl, Reglergrenzen, Permalink), **Footer**, **Laufzeit-Importe**.
* Testzahl und Laufzeit: siehe „Tests ausführen“.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Seitenleiste, Kernfragen, vier Experimente, 🚧 Grenzen, 📐 Formeln |
| `erv_constants.py` | Modellklassen, Seeds, Regler-Grenzen, Beispielszenarien, Farben |
| `erv_generator.py` | Kopie des Generators der Verweilzeit (Funktionen und Konstanten wortgleich) |
| `erv_rng.py` | Kopie der SplitMix64-Vorlage (ohne `math.log` und `math.cos` der C-Bibliothek) |
| `erv_exact.py` | Wertetabelle und exakte Aufzählung (Koalitionsformel) |
| `erv_sampler.py` | Stichproben-Erklärer auf der Wertetabelle (NumPy-Zufallsstrom = SplitMix64) |
| `erv_rank.py` | Rangkorrelation, relativer L2, Rang 1, Wichtigkeit, Art-Anteil, Paarmaße |
| `erv_schichten.py` | Schichten Daten, Modell, Erklärer; Spitzenmerkmale; Wichtigkeit gegen Wahrheit |
| `erv_wahrheit.py` | Wahrheit je Funktional aus der exakten Bevölkerung (1224 Stützpunkte) |
| `erv_tables.py` | Lesen der eingefrorenen Daten |
| `erv_evaluation.py` | Auswertung der Experimente und alle Verdikte als reine Funktionen |
| `erv_presets.py` | Permalink (Begrenzen), Beispielszenarien (Schnellstart in der Seitenleiste, aktives hervorgehoben), Zufalls-Seed-Knopf |
| `erv_texts.py` | feste Texte (Einleitung, Grenzen, Formeln) |
| `erv_visualization.py` | Plotly-Abbildungen (Achsen fest) |
| `erv_layout.py` | fester Kopf (Stylesheet) |
| `data/` | eingefrorene Beiträge (`erv_phi*.npz`), Wertetabellen (`erv_tables.npz`), Schicht Erklärer (`erv_erklaerer*.json`), exakte Bevölkerung (`erv_wahrheit.json`), `QUELLE.json` |
| `tools/erv_make_tables.py` | erzeugt und prüft die eingefrorenen Daten (mit scikit-learn, `--write`, `--check`) |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Randfälle, Verdikte, Claims, eingefrorene Daten, Kopien, Presets, Layout, App, Footer, Laufzeit-Importe |
| `requirements.txt` / `requirements-dev.txt` | Laufzeit (streamlit, numpy, plotly, pandas) / plus pytest, scipy, scikit-learn |
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |

## 🚧 Bewusst nicht enthalten / nicht gemessen

* Andere Wertfunktionale (konditional, pfadabhängig, Leave-one-out), KernelSHAP als Erklärer, Mengen gleich guter Modelle (Rashomon) und der Wert eines Trainingspunkts (Datenwert): eigene Demos der Reihe, hier nur Verweis.
* Intervalle und ihre Abdeckung; Zeilenmengen über 20 Zeilen; andere Hintergrundgrößen; mehr als sechs Merkmale; ein schwächer regularisiertes Netz.
* Der Nachbau des LIME-artigen Surrogats (nur zitiert), Menschenstudien, Nutzen in Betrieb, Logistik oder Recht.
* PDF-Export.

## Quellen

Stufe W = im Volltext gelesen; A = nur Zusammenfassung bzw. Posterseite; Preprints aus 2026 sind unbegutachtet.

| Quelle | Stufe | Verwendung |
|---|---|---|
| Štrumbelj, Kononenko (2010), *An Efficient Explanation of Individual Classifications using Game Theory*, JMLR 11, 1 bis 18 | W | Permutationsform des Stichproben-Erklärers (Gl. 7); Stichprobenzahl m = Z²σ²/e² (etwa 65 000 laut Quelle; 66 349 nachgerechnet in der Planung, hier nicht neu gerechnet) |
| Hwang u. a. (2026), „Explanation Multiplicity“, NeurIPS-2026-Poster | A (Posterseite) | wiederholte Läufe bei festem Modell; L2-Abstand verdeckt Rangänderungen. Hier nachgemessen, **nicht übernommen**; nicht im Volltext gelesen |
| Covert, Lee (2021), AISTATS | A | Schätzfehler und Varianz bei KernelSHAP (nur Einordnung) |
| Krishna u. a. (2024), TMLR | A | Erklärverfahren stimmen oft nicht überein (nur Einordnung) |
| Rahnama u. a. (2021) | A | Rangfolge der Erklärer hängt von Modell, Datenpunkt, Normierung und Maß ab (nur Einordnung) |

**Keine Neuheitsaussage:** Neu-Training und Erklärer-Zufall sind Forschung (Vorbilder: Fisher/Rudin/Dominici 2019, Hwang u. a. 2026; Wahrheits-Benchmarks: XAI-Bench, OpenXAI, Hooker/Mentch). Neu ist nur die nachrechenbare Vorführung mit Zähleinheit, Orakel gegen Aufzählung und bekannter Wahrheit je Funktional.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Die Suite hat **145 Tests**: 16 Handbeispiele, 17 Orakel, 6 Eigenschaften, 11 Randfälle, 9 Verdikte, 19 Claims, 12 zu den eingefrorenen Daten, 6 Kopie-Tests, 33 zu Presets und Layout, 7 App-Tests, 2 Footer-Tests, 1 Laufzeit-Import-Test und 6 Text-Tests. Laufzeit lokal (Windows, Python 3.14, seriell, Maschine durch andere Läufe belastet): 145 s am 2026-11-20.

Eingefrorene Daten neu erzeugen oder prüfen (scikit-learn nötig; die Neurechnung dauert je Satz einige Minuten): `python tools/erv_make_tables.py --write` bzw. `--check`.

Gebaut mit Streamlit, Plotly, NumPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
