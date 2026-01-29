# VWL-Simulation-RL

Multi-Agent Reinforcement Learning Volkswirtschaftssimulation mit PPO für DHSH KI-Projekt.

## Installation

### Voraussetzungen

- Python 3.11 oder höher
- pip (Python Package Manager)

### Setup

1. **Repository klonen:**

```bash
git clone https://github.com/H3nri5H/VWL-Simulation-RL.git
cd VWL-Simulation-RL
```

2. **Virtuelle Umgebung erstellen und aktivieren:**

```bash
python3.11 -m venv venv

# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

3. **Abhängigkeiten installieren:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bibliotheken

Das Projekt verwendet folgende Hauptbibliotheken:

- **Ray/RLlib 2.10+**: Multi-Agent Reinforcement Learning Framework
- **Gymnasium**: Environment API
- **PettingZoo**: Multi-Agent Environment Wrapper
- **NumPy**: Numerische Berechnungen
- **Matplotlib**: Visualisierung

### Training starten

```bash
python train.py
```

## Projektstruktur

```
VWL-Simulation-RL/
├── env/                  # Environment Implementierung
│   ├── __init__.py
│   └── economy_env.py    # Volkswirtschafts-Simulation
├── train.py              # Training-Script
├── requirements.txt      # Python-Abhängigkeiten
├── CHANGELOG.md          # Änderungsdokumentation
└── README.md             # Diese Datei
```

## Lizenz

DHSH Fortgeschrittene KI-Anwendungen - Studienprojekt
