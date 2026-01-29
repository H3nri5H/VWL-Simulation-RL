# Changelog

Diese Datei dokumentiert alle wichtigen Änderungen am Projekt. Jeder Commit wird hier mit einer kurzen Beschreibung aufgeführt.

## Format

```
## [Datum] - Commit-Hash
### Beschreibung
- Was wurde geändert/hinzugefügt/entfernt
```

---

## [2026-01-29] - cf9dc0b
### Test-Script hinzugefügt
- `test_env.py` erstellt für manuelles Environment-Testing
- **Funktioniert ohne Ray/TensorFlow** - nur pure Python!
- Simuliert 20 Steps mit zufälligen Actions
- Zeigt Wirtschafts-State, Preise, Löhne, Profite pro Step
- Nutze: `python test_env.py`

## [2026-01-29] - 107996a
### Economy Environment vollständig implementiert
- **Firmen-State**: Preis, Lohn, Inventory, Kapital, Profit, Nachfrage
- **Haushalts-Logik** (regelbasiert):
  - Arbeiten bei Firma mit höchstem Lohn
  - Kaufen bei billigster Firma (80% des Geldes ausgeben)
- **Markt-Clearing pro Step**:
  1. Firmen passen Preise/Löhne an (via Actions)
  2. Haushalte wählen Arbeitgeber (höchster Lohn)
  3. Haushalte verdienen Lohn
  4. Haushalte kaufen Güter (billigste Firma zuerst)
  5. Firmen produzieren (10 Einheiten pro Mitarbeiter)
  6. Profit-Berechnung: Umsatz - Lohnkosten
- **Reward**: Normalisierter Profit (Mean-Centered, Std-Scaled)
- **Observations**: [Preis, Lohn, Nachfrage, Inventory, Profit, Markt-Avg-Preis, Markt-Avg-Lohn]
- **`render()`** Methode zeigt Wirtschafts-State im Terminal
- Max 100 Steps pro Episode (25 Jahre bei Quartals-Steps)

## [2026-01-29] - d4feb28 + 42aa8ed
### Framework-Wechsel: PyTorch → TensorFlow
- `train.py`: Framework von 'torch' zu 'tf2' geändert
- `requirements.txt`: `torch` durch `tensorflow` ersetzt
- **Grund**: PyTorch hat DLL-Probleme auf Windows (c10.dll WinError 1114)
- TensorFlow hat bessere Windows-Kompatibilität

## [2026-01-29] - c60feee
### PyTorch hinzugefügt
- `torch` zu requirements.txt hinzugefügt
- **Grund**: RLlib braucht ein Deep Learning Framework (PyTorch oder TensorFlow)
- `train.py` nutzt `.framework('torch')` → PyTorch muss installiert sein
- Behebt ImportError beim Training-Start

## [2026-01-29] - 1a65e51
### Dependency-Vereinfachung
- Unnötige Version-Constraints entfernt (NumPy, Pandas, Matplotlib, etc.)
- Nur noch **kritische** Versionen fixiert:
  - `ray[rllib]>=2.10.0` (RLlib API-Stabilität)
  - `pyarrow<21.0.0` (Ray-Kompatibilität)
- **Vorteil**: pip löst Dependencies automatisch, weniger Konflikte
- **Prinzip**: "So wenig Version-Constraints wie möglich, so viel wie nötig"

## [2026-01-29] - 75d87e0
### PyArrow Kompatibilitäts-Fix
- PyArrow Version Constraint `>=6.0.0,<21.0.0` hinzugefügt zu requirements.txt
- **Problem**: Ray 2.10.0 nutzt `PyExtensionType`, das in PyArrow 21.0.0 entfernt wurde
- **Lösung**: PyArrow auf Version < 21.0.0 beschränkt
- Behebt Fehler: `module 'pyarrow' has no attribute 'PyExtensionType'`

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
