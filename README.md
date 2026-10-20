# 🤖 Roboterprogrammierung – Path Planning

Dieses Repository ist mein persönlicher Fork der Vorlesungsunterlagen zur **Roboterprogrammierung (Introduction to Robot Path Planning)** von Prof. Björn Hein.

---

## 🔄 1. Git-Workflow im Semester

Damit der `main`-Branch sauber bleibt und Updates des Dozenten problemlos eingespielt werden können, wird `main` nur für offizielle Updates genutzt und eigene Bearbeitungen finden in separaten Branches statt.

*(Einmalig, falls noch nicht geschehen: Original-Repo als `upstream` verknüpfen mit `git remote add upstream <ORIGINAL-REPO-URL>`)*

### Updates des Dozenten abrufen
```powershell
git switch main          # Auf den Haupt-Branch wechseln
git fetch upstream       # Änderungen vom Original-Repo laden
git merge upstream/main  # In den lokalen main einpflegen
git push origin main     # Eigenen GitHub-Fork aktualisieren
```

### Aufgaben bearbeiten & sichern
```powershell
# 1. Neuen Branch für ein Kapitel / eine Übung anlegen
git switch -c uebung-01

# 2. Änderungen lokal speichern (Commit)
git add .
git commit -m "Lösung Übung 1 fertiggestellt"

# 3. Backup auf eigenen GitHub-Fork hochladen
git push -u origin uebung-01   # Beim ersten Mal im Branch, danach reicht: git push
```

---

## 📚 2. Projektübersicht & Kapitel

In diesem Projekt geht es um **Roboter-Pfadplanung (Path Planning)** in Python mittels Jupyter Notebooks. Ziel ist es, kollisionsfreie Pfade für verschiedene Robotersysteme in Hindernisumgebungen zu berechnen, zu visualisieren und zu vergleichen.

### ⚙️ Setup & Voraussetzungen
* **Python:** `>= 3.11`
* **Installation der Pakete:**
  ```powershell
  pip install -r requirements.txt
  ```
  *(Wichtigste Bibliotheken: `numpy`, `scipy`, `networkx`, `shapely`, `matplotlib`, `jupyter`)*

### 🗂️ Kapitel & Notebooks (`notebooks/`)

| Kapitel | Notebook(s) | Thema / Inhalt |
| :--- | :--- | :--- |
| **01 – Einführung** | `IP-1-0-Introduction.ipynb` | Übersicht, Ressourcen und benötigte Python-Bibliotheken |
| **02 – Grundlagen** | `IP-2-0-Pathplanning-Introduction.ipynb` | Einführung in die Konzepte der Pfadplanung und des Konfigurationsraums |
| **03 – Graphen (NetworkX)** | `IP-3-0-Introduction-to-NetworkX.ipynb`<br>`IP-3-1-Exercises-with-NetworkX_empty.ipynb` | Arbeiten mit Graphen in Python (`networkx`) inkl. praktischer Übungsaufgaben |
| **04 – Testumgebungen** | `IP-4-0-Test-Environments.ipynb` | Definition von Hindernissen, Arbeitsräumen und Kollisionstests (`shapely`) |
| **05 – PRM (Probabilistic Roadmaps)** | `IP-5-0-PRM_Basics.ipynb`<br>`IP-5-1-PRM-Basics-class.ipynb`<br>`IP-5-2-PRM-KClosest.ipynb` | Grundprinzip von PRM, objektorientierte Implementierung und Nachbarschaftssuche (*k-Closest*) |
| **06 – Visibility PRM** | `IP-6-0-PRM-Visibility.ipynb` | Kompaktere Roadmaps durch Sichtbarkeits-Kriterien (Guards & Connectors) |
| **07 – Lazy PRM** | `IP-7-0-PRM-Lazy.ipynb` | Verzögerte Kollisionsprüfung zur Beschleunigung der Planung |
| **08 – A\*-Algorithmus** | `IP-8-0_ASTAR.ipynb` | Optimale Pfadsuche auf Graphen und Gitternetzen mittels Heuristiken |
| **09 – RRT (Rapidly-exploring Random Trees)** | `IP-9-0-RRT-Basics.ipynb` | Baum-basierte Zufallsplanung für schnelle Exploration im Raum |
| **10 – Kinematik & Roboterformen** | `IP-10-0-PlanarManipulator.ipynb`<br>`IP-10-1-PlanarManipulatorTests.ipynb`<br>`IP-10-2-ShapeRobot.ipynb` | Planung für mehrgliedrige Roboterarme (Planar Manipulator) und ausgedehnte 2D-Roboterformen |
| **11 – Bridge-Builder (BB)** | `IP-11-0-BB-Method.ipynb` | Spezielle Sampling-Strategie für enge Passagen (Narrow Passages) |
| **X – Benchmarking & Tests** | `IP-X-0-Benchmarking-concept.ipynb`<br>`IP-X-1-Automated_PlanerTest.ipynb` | Automatisierte Auswertung, Laufzeit- und Erfolgsvergleiche der verschiedenen Planer |

### 🧩 Python-Module im Hintergrund
Neben den Notebooks liegen im Ordner `notebooks/` die zugehörigen Python-Implementierungen:
* **`IPPlanerBase.py` / `IPPRMBase.py` & Algorithmen (`IPBasicPRM.py`, `IPRRT.py`, `IPAStar.py`, ...):** Kernklassen der jeweiligen Planungsverfahren.
* **`IPEnvironment*.py` & `IPPlanarManipulator.py`:** Umgebungsmodelle, Kinematik und Kollisionsprüfung.
* **`IPVIS*.py`:** Hilfsmodule zur interaktiven Visualisierung und Animation der Suchbäume und Pfade.
* **`IPBenchmark.py` / `IPTestSuite.py` / `IPPerfMonitor.py`:** Werkzeuge zur Performance-Messung und Evaluierung.

