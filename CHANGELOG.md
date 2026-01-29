# Changelog

Diese Datei dokumentiert alle wichtigen Änderungen am Projekt. Jeder Commit wird hier mit einer kurzen Beschreibung aufgeführt.

## Format

```
## [Datum] - Commit-Hash
### Beschreibung
- Was wurde geändert/hinzugefügt/entfernt
```

---

## [2026-01-29] - 80f445e
### Dependency Fix
- Gymnasium Version Constraint von `==0.29.1` zu `>=0.29.0` geändert
- Vermeidet Installationskonflikte mit verschiedenen Python-Versionen und abhängigen Packages
- Installation ist jetzt flexibler und robuster

## [2026-01-29] - 49d31d9
### Action Space Verbesserung
- Action Space von `Discrete(9)` zu `MultiDiscrete([5, 5])` geändert
- Firmen können jetzt **separat** über Preis und Lohn entscheiden:
  - Dimension 0 (Preis): 0=-10%, 1=-5%, 2=0%, 3=+5%, 4=+10%
  - Dimension 1 (Lohn): 0=-10%, 1=-5%, 2=0%, 3=+5%, 4=+10%
- `_decode_action()` Hilfsmethode hinzugefügt zur Konvertierung von Actions zu Prozentänderungen
- Docstring in `economy_env.py` aktualisiert

## [2026-01-29] - 63b61ed
### Projekt-Setup
- Repository erstellt
- Grundlegende Projektstruktur angelegt:
  - `env/` Verzeichnis für Environment-Implementierung
  - `train.py` als Training-Script-Platzhalter
- `README.md` mit Installationsanleitung erstellt
- `requirements.txt` mit Abhängigkeiten (Ray/RLlib 2.10.0, Gymnasium, PettingZoo, NumPy, Matplotlib)
- `CHANGELOG.md` für Änderungsdokumentation angelegt
- `.gitignore` für Python-Projekte hinzugefügt
- Basis-Skeleton für `economy_env.py` erstellt
