# IP-7-1 Lab: Maze & Path Planning – Notizen & Zusammenfassung

Diese Notizen fassen die Konzepte, Aufgaben und Erkenntnisse aus dem Labor **IP-7-1-Lab-Maze** (`IP-7-1-Lab-Maze_solution.ipynb`) zusammen.

---

## 1. Umgebung, Benchmark-Validierung & Performance-Monitoring (Task 1)

### Wichtige Klassen & Module
* **`Maze`** (`IPMaze.py`): Erzeugt ein zufälliges 2D-Labyrinth (standardmäßig $6 \times 6$ Zellen auf $[0, 22] \times [0, 22]$ mit Wandstärke `0.3`). Jede Wand ist ein `shapely.geometry.box`-Polygon.
* **`CollisionChecker`** (`IPEnvironment.py`): Kapselt die Hindernisse (`scene`) und die Raumgrenzen (`limits`) und bietet Methoden zur Kollisionsprüfung:
  * `pointInCollision(pos)`: Prüft, ob ein Punkt `shapely.Point(pos)` ein Hindernis schneidet (`intersects`).
  * `lineInCollision(startPos, endPos, steps=40)`: Diskrete Abtastung einer Verbindungsstrecke mit `steps` Zwischenpunkten (ruft intern `pointInCollision` auf).
  * `distance(pos)`: Berechnet den minimalen euklidischen Abstand eines Punktes zum nächsten Hindernis (Freiraum / *Clearance*).
* **`Benchmark`** (`IPBenchmark.py`): Bündelt Name, `CollisionChecker`, `startList` (Startkonfiguration), `goalList` (Zielkonfiguration), Beschreibung und Schwierigkeitsgrad (`level`).
* **`IPPerfMonitor`**: Dekoriert Methoden (wie `pointInCollision` und `lineInCollision`) mit `@IPPerfMonitor` und zeichnet jeden Funktionsaufruf samt Argumenten (`args`), Rückgabewert (`retval`) und Laufzeit (`time`) auf. Auswertung erfolgt als `pandas.DataFrame` über `IPPerfMonitor.dataFrame()`.

### Erkenntnisse aus Task 1a & 1b
* **Validierung (`validateBenchmark`):**
  * Prüft, ob es sich um einen 2D-Raum handelt (`dim == 2`) und ob für beide Achsen gültige Grenzen (`low < high`) existieren.
  * Prüft, ob `start` und `goal` innerhalb der Raumgrenzen liegen und **kollisionsfrei** sind (`not pointInCollision(pos)`).
  * Gibt über `checker.distance(pos)` den Sicherheitsabstand (*Clearance*) von Start und Ziel zur nächsten Wand aus (im Labyrinth jeweils `1.10`, da eine Zelle $3.0$ breit ist, die Wand $0.3$ dick ist und der Punkt mittig bei $\frac{3.0 - 0.3}{2} - 0.25 = 1.10$ liegt).
* **Zufällige Punktprüfungen (2000 Samples):**
  * Die gemessene Kollisionsrate ($\approx 17\text{--}18\,\%$) entspricht dem Flächenverhältnis $\frac{\text{Hindernisfläche}}{\text{Gesamtfläche}}$ im Konfigurationsraum.

---

## 2. Kollisionsprüfungen visualisieren: `BasicPRM` vs. `LazyPRM` (Task 2)

### Funktionsweise von `drawCollisionChecks(calls, ax)` (Task 2a)
1. Filtert das Log-DataFrame auf `calls["name"] == "pointInCollision"`.
2. Extrahiert die $(x, y)$-Koordinaten aus `args[1]` (da `args[0]` die `self`-Referenz des `CollisionChecker` ist) und das Prüfergebnis (`True`/`False`) aus `retval`.
3. Entfernt mehrfach geprüfte identische Punkte mit `.drop_duplicates()`, damit beim Plotten keine Punkte übereinander gezeichnet werden.
4. Zeichnet freie Punkte grün (`FREE_COLOR`) und kollidierende Punkte rot (`COLLISION_COLOR`) als vektorisierten `ax.scatter`-Plot.

### Vergleich `BasicPRM` vs. `LazyPRM` (Task 2b & 2c)

| Eigenschaft | `BasicPRM` (`IPBasicPRM.py`) | `LazyPRM` (`IPLazyPRM.py`) |
| :--- | :--- | :--- |
| **Strategie beim Aufbau** | Prüft jeden zufälligen Knoten und jede Kante **sofort** beim Erstellen der Roadmap. | Baut die komplette Roadmap zunächst **ohne Kollisionsprüfung** auf (*„lazy"* = faul). |
| **Räumliche Verteilung der Checks** | **Gleichmäßig im gesamten Raum** verteilt (auch in Sackgassen, die für den Pfad irrelevant sind). | **Stark fokussiert** entlang vielversprechender Kandidatenpfade zwischen Start und Ziel. |
| **`pointInCollision`-Aufrufe (Maze)** | ca. $210.000$ | ca. $509.000$ (bei nur $\approx 22.000$ unterschiedlichen Punkten!) |

### Warum hat `LazyPRM` über $500.000$ Punkt-Checks bei nur $\approx 22.000$ eindeutigen Punkten?
* `LazyPRM` sucht in jeder Iteration den kürzesten Pfad im ungeprüften Graphen und prüft diesen in `_checkForCollisionAndUpdate` schrittweise vom Start Richtung Ziel (zuerst alle Knoten, dann alle Kanten).
* Sobald weit hinten auf dem Kandidatenpfad ein Knoten oder eine Kante kollidiert, wird **nur dieses eine kollidierende Element** aus der Roadmap gelöscht und die Prüfung sofort abgebrochen.
* **Das Problem:** `LazyPRM` merkt sich nicht, dass die vorderen Knoten und Kanten dieses Pfades bereits erfolgreich als **frei** getestet wurden! Im nächsten Durchlauf beginnt der neue Kandidatenpfad fast immer mit denselben vorderen Knoten/Kanten und prüft diese wieder und wieder (im Schnitt wird jeder Punkt über 20-mal geprüft).

---

## 3. Exakte Validierung, Scheibenroboter & `LazierPRM` (Task 3)

### Task 3a: Exakte Pfadvalidierung (`validatePath`)
* **Problem von `lineInCollision`:** Da Kanten standardmäßig nur an diskreten Zwischenpunkten (`steps`) abgetastet werden, können lange Kanten bei zu kleiner Schrittzahl (`steps=5`) dünne Wände (`thinBenchmark`) einfach „überspringen". Der Planer hält den Pfad dann fälschlicherweise für frei.
* **Lösung (`validatePath`):** Nutzt `lineInCollisionExact(u, v)`, welches intern ein echtes `shapely.geometry.LineString([u, v])`-Objekt erzeugt und geometrisch exakt auf Schnitt mit den Hindernispolygonen prüft.
* Erst mit ausreichend feiner Diskretisierung (`steps=100`) findet `BasicPRM` im `thinBenchmark` einen Pfad, der auch die exakte Prüfung `validatePath` besteht (`True`).

### Task 3b: Scheibenroboter (`DiscCollisionChecker`)
* Ein kreisförmiger Roboter mit Radius $r > 0$ im Arbeitsraum ($\mathcal{W}$-Space) wird im Konfigurationsraum ($\mathcal{C}$-Space) wieder auf einen **Punktroboter** reduziert, indem man alle Hindernisse um den Radius $r$ **aufbläst** (Minkowski-Summe mit Kreis vom Radius $r$).
* In Shapely gelingt dies direkt beim Initialisieren über `obstacle.buffer(radius)`.
* **Verhalten bei steigendem Radius im Labyrinth:**
  * **$r = 0.5$ (`path found: True`):** Die Gänge (lichte Weite $2.7$) sind noch breit genug ($\ge 2 \cdot 0.5 = 1.0$).
  * **$r = 1.0$ (`path found: False`):** Ein Scheibenroboter mit Radius $1.0$ benötigt mindestens $2.0$ Platz, für diagonale Bewegungen und Kurven um Ecken bzw. wegen der diskreten Abtastung reicht der verbleibende Korridor an Engstellen nicht mehr für eine zusammenhängende PRM-Verbindung.
  * **$r = 1.2$ (`start/goal in collision!`):** Start und Ziel haben im Labyrinth einen Wandabstand (*Clearance*) von `1.10`. Für $r = 1.2 > 1.10$ liegen Start und Ziel bereits innerhalb der aufgeblasenen Hindernisse!

### Task 3c ★: `LazierPRM` (Caching freier Knoten & Kanten)
* **Idee:** Wir erweitern `LazyPRM` um zwei Mengen (`set`):
  * `self.freeNodes`: Speichert alle Knoten, die bereits mit `pointInCollision == False` getestet wurden.
  * `self.freeEdges`: Speichert alle ungerichteten Kanten `(min(u, v), max(u, v))`, die bereits mit `lineInCollision == False` getestet wurden.
* **Wichtig:** Zu Beginn jedes `planPath`-Aufrufs müssen `self.freeNodes.clear()` und `self.freeEdges.clear()` aufgerufen werden, falls dieselbe Planer-Instanz mehrfach verwendet wird.
* **Ergebnis:**
  * Der gefundene Pfad ist **exakt identisch** zu `LazyPRM` (`lazierSolution == solution`), da genau dieselben kollidierenden Elemente in derselben Reihenfolge entfernt werden.
  * Die Anzahl der `pointInCollision`-Aufrufe sinkt von **509.239** auf **24.646** – eine Reduktion um **ca. $95\,\%$**!

---

## 4. Deterministischer `BugPlanner` (Task 4)

### Funktionsweise (Task 4a)
Der `BugPlanner` bewegt sich auf einem 4-Nachbarschafts-Gitter mit Schrittweite `stepSize = 0.5` in den Richtungen Rechts (`0`), Oben (`1`), Links (`2`) und Unten (`3`). Er wechselt zwischen zwei Modi:

1. **Leave-Bedingung (immer zuerst prüfen):**
   * Ist die direkte Sichtlinie von der aktuellen Position `pos` zum `goal` komplett frei (`not lineInCollisionExact(pos, goal)`), verbindet der Planer den aktuellen Knoten direkt mit `"goal"` und gibt den kürzesten Pfad im aufgebauten Laufgraphen zurück (`nx.shortest_path(self.graph, "start", "goal")`).
2. **Schleifen-Erkennung (Loop Detection):**
   * Speichert jedes Tupel `(i, j, direction)` in einem Set `visited`. Trifft der Roboter im selben Zustand erneut auf ein bereits besuchtes Gitterfeld mit gleicher Blickrichtung, befindet er sich in einer Endlosschleife und bricht mit `return []` ab.
3. **Modus 1 – Geradeaus laufen (`followingWall == False`):**
   * Versucht in aktueller `direction` einen Schritt vorwärts zu machen.
   * **Hit-Bedingung:** Ist der Schritt durch eine Wand blockiert, dreht sich der Roboter um $90^\circ$ nach links (`direction = (direction + 1) % 4`) und aktiviert den Wandfolgemodus (`followingWall = True`).
4. **Modus 2 – Wand folgen mit der Rechten-Hand-Regel (`followingWall == True`):**
   * Da der Roboter beim Auftreffen auf die Wand nach links gedreht hat, liegt die Wand rechts von ihm.
   * Um die Wand rechts zu behalten, prüft er die 4 Richtungen in genau dieser Priorität:
     1. **Rechts** `(direction - 1) % 4` (um Außenecken der Wand herum),
     2. **Geradeaus** `direction` (weiter an der geraden Wand entlang),
     3. **Links** `(direction + 1) % 4` (Innenecke),
     4. **Zurück** `(direction + 2) % 4` (Sackgasse).

### Erkenntnisse aus Task 4b & 4c
* **Ohne Leave-Bedingung:** Der Bug verlässt den Wandfolgemodus nie mehr, sobald er einmal auf ein Hindernis gestoßen ist. Er läuft endlos im Kreis um die Hindernisstruktur herum, bis die `visited`-Schleifenerkennung abbricht.
* **`lineInCollision` (sampled) statt `lineInCollisionExact` in der Leave-Bedingung:**
  * Da die Sichtlinie vom aktuellen Punkt zum Ziel diagonal durch das gesamte $22 \times 22$-Labyrinth reichen kann (Länge bis $\approx 25$), hat `lineInCollision` mit `steps=40` einen Punktabstand von $\approx 0.6 > \text{Wandstärke } 0.3$.
  * Die Sichtlinie überspringt dünne Wände, der Bug zieht fälschlicherweise eine direkte Linie mitten durch Wände zum Ziel, und `validatePath` meldet einen **ungültigen Pfad** (`False`).

---

## 5. Systematisches Benchmarking (Task 5)

### Vergleich über alle Benchmarks (`RUNS = 3`)
* **Welcher Planer ist am besten?**
  * Es gibt keinen einzelnen Planer, der überall dominiert (*No Free Lunch*):
  * **`BugPlanner`:** Extrem schnell und sparsam bei Punktprüfungen in einfachen Umgebungen und bei der klassischen U-Falle (`Trap`), in der Sampling-Planer mit kleinen Radien oft Probleme haben. Dafür erzeugt er deutlich längere, stufenförmige Pfade und scheitert an komplexeren Hindernissen ohne direkte Sichtlinie zum Ziel (z. B. `Maze (seed 1)` oder isolierte Inseln).
  * **`BasicPRM`:** Sehr hohe Erfolgsquote über fast alle Benchmarks und deutlich kürzere, glattere Pfade als der Bug-Planer, benötigt aber konstant viele Kollisionsprüfungen für den kompletten Roadmap-Aufbau.
  * **`LazyPRM`:** Findet ähnlich kurze Pfade wie `BasicPRM`. In einfachen Szenen mit wenig Hindernissen zwischen Start und Ziel ist er extrem effizient (sehr wenige Punktprüfungen). In verwinkelten Labyrinthen (`Maze`) explodiert jedoch die Anzahl der Punktprüfungen ohne Caching (wie in `LazierPRM`), da ständig Kandidatenpfade verworfen und neu geprüft werden.
* **Warum reicht 1 Durchlauf für `BugPlanner`, aber nicht für PRM?**
  * **`BugPlanner` ist vollständig deterministisch:** Bei gleicher Umgebung liefert er in jedem Lauf exakt denselben Pfad und dieselbe Anzahl an Checks (Varianz $= 0$).
  * **`BasicPRM` & `LazyPRM` sind probabilistisch (stichprobenbasiert):** Da die Knoten zufällig gezogen werden (`random`), schwanken Erfolgsrate, Pfadlänge und Prüfaufwand von Lauf zu Lauf. Erst mehrere Läufe (`RUNS >= 3`) liefern statistisch belastbare Mittelwerte.
* **Warum sind Bug-Pfade so lang und eckig – und wie verbessert man sie?**
  * Der Bug läuft erst stur in eine Achsenrichtung, folgt dann achsenparallel allen Umwegen einer Wand und biegt erst bei komplett freier Sichtlinie zum Ziel ab. Zudem erlaubt das 4er-Gitter keine Diagonalen (Manhattan-Metrik: $1 + 1 = 2$ statt $\sqrt{2} \approx 1.41$).
  * **Verbesserung:** Nachgelagertes **Path Shortcutting / Smoothing** (iterativ zwei nicht-benachbarte Wegpunkte wählen und direkt verbinden, falls `lineInCollisionExact` frei ist) oder Nutzung eines 8-Nachbarschafts-Gitters bzw. von *Bug1 / Bug2 / TangentBug* (direkt auf das Ziel zulaufen statt erst entlang einer Koordinatenachse).

---

## 6. Eigene Benchmark-Umgebung: `Island trap` (Task 6)

### Aufbau von `islandBenchmark` (Task 6a & 6b)
Um eine Umgebung zu konstruieren, in der **`BugPlanner` scheitert**, während **`BasicPRM` und `LazyPRM` zuverlässig gewinnen**, nutzen wir die lokale Natur des Bug-Planers aus:

* **Start:** `[3.0, 11.0]` (links)
* **Ziel:** `[19.0, 11.0]` (rechts)
* **Hindernis 1 – Die „Insel" (`box(6.0, 8.0, 8.5, 14.0)`):** Liegt direkt vor dem Startpunkt. Der `BugPlanner` läuft nach rechts, trifft auf diese Insel und beginnt sie im Wandfolgemodus zu umrunden.
* **Hindernis 2 – Die Abschirmwand (`box(12.5, 3.0, 14.5, 19.0)`):** Eine breite vertikale Wand zwischen der Insel und dem Ziel. Sie verhindert, dass der `BugPlanner` von irgendeinem Punkt auf dem Rand der Insel jemals eine freie Sichtlinie (`lineInCollisionExact`) zum Ziel bekommt!

### Ergebnis auf `Island trap`
1. **`BugPlanner` (`path found: False`):** Trifft auf die Insel, umrundet sie komplett, erhält wegen der Abschirmwand an keiner Stelle freie Sicht zum Ziel (Leave-Bedingung wird nie wahr) und bricht nach einer vollen Runde durch die `visited`-Schleifenerkennung ab.
2. **`BasicPRM` & `LazyPRM` (`path found: True`):** Da beide Planer Knoten global im gesamten freien Raum verteilen, überbrücken ihre Roadmaps problemlos den freien Zwischenraum zwischen Insel und Abschirmwand sowie die Durchgänge oben ($y > 19$) und unten ($y < 3$).
