# AGENTS.md – Arbeitsregeln für dieses Repository

Dieses Repository ist ein persönlicher Fork der Vorlesung **Roboterprogrammierung (Introduction to Robot Path Planning)** (`upstream`: `https://github.com/bjoernhein/intro-path-planning.git`). Die folgenden Regeln gelten rechnerübergreifend für alle KI-Assistenten.

## Sprache & Stil
* **Sprache:** Antworten, Erklärungen und neue Code-Kommentare grundsätzlich auf **Deutsch** (kurz, präzise und verständlich).
* **Bestehende Inhalte erhalten:** Vorhandene `# TODO`-Kommentare, Docstrings und eigene Notizen in Notebooks **niemals entfernen**, außer es wird ausdrücklich gewünscht.
* **Lernmodus:** Wenn nach Hinweisen gefragt wird, nur Denkanstöße geben und nicht die fertige Lösung vorwegnehmen.

## Umgang mit Code & Notebooks
* **Merge-Konflikte mit `upstream` vermeiden:** Die Basis-Module des Dozenten in `notebooks/` (z. B. `IPAStar.py`, `IPEnvironment.py`, `IPPlanerBase.py`, `IPBasicPRM.py`, `IPLazyPRM.py`) möglichst **nicht direkt ändern**. Erweiterungen und Experimente stattdessen als Unterklasse oder eigene Funktion direkt im jeweiligen Notebook umsetzen.
* **Arbeitsverzeichnis für Notebooks:** Notebooks und Python-Skripte immer aus `notebooks/` heraus ausführen, damit relative Modul-Imports (`from IPAStar import AStar` usw.) funktionieren.
* **Notebooks verifizieren:** Nach Code-Änderungen in einem Notebook dieses bei Bedarf mit `python -m jupyter nbconvert --to notebook --execute --inplace <Notebook>.ipynb` aus `notebooks/` heraus ausführen und auf Fehlerfreiheit prüfen.
* **Notebook-Format:** Keine unnötigen neuen Metadatenfelder (wie `id` in älteren `nbformat 4.2`-Notebooks) beim programmatischen Bearbeiten einfügen.
* **Rechnerunabhängigkeit:** Keine absoluten/lokalen Pfade fest verdrahten – immer relative Pfade ab Repository-Root (`notebooks/...`) verwenden.

## Projektstruktur & Dokumentation
* Alle Vorlesungs-Notebooks und Python-Hilfsmodule liegen in `notebooks/`.
* Abhängigkeiten stehen in `requirements.txt` (Python `>= 3.11`).
* Bei neuen Notebooks oder Notizdateien die Übersicht in `README.md` aktuell halten.
