# VWL-Simulation-RL

Multi-Agent Reinforcement Learning Volkswirtschaftssimulation mit PPO für DHSH KI-Projekt.

## 🚀 Quick Start

### Option 1: Environment testen (ohne Training)

```bash
git clone https://github.com/H3nri5H/VWL-Simulation-RL.git
cd VWL-Simulation-RL
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
python test_env.py
```

### Option 2: Dashboard starten (Visualisierung)

**Terminal 1 - Backend:**
```bash
# Windows:
start_backend.bat

# Linux/Mac:
chmod +x start_backend.sh
./start_backend.sh
```

**Terminal 2 - Frontend:**
```bash
# Windows:
start_frontend.bat

# Linux/Mac:
chmod +x start_frontend.sh
./start_frontend.sh
```

Dashboard öffnet sich automatisch auf http://localhost:8501

### Option 3: Training (nur auf Linux/WSL2/Cloud)

```bash
python train.py
```

⚠️ **Hinweis**: Training funktioniert aktuell nicht nativ auf Windows (DLL-Probleme mit Ray/PyTorch/TensorFlow). Nutze WSL2, Google Cloud oder Google Colab.

## 📚 Features

### ✅ Implementiert

- **Economy Environment** (`env/economy_env.py`):
  - 2+ Firmen (KI-gesteuert)
  - 10+ Haushalte (regelbasiert)
  - Markt-Clearing: Preise, Löhne, Nachfrage, Angebot
  - Rewards: Profit-basiert

- **Testing** (`test_env.py`):
  - Manuelles Environment-Testing ohne Ray
  - Funktioniert auf Windows!

- **Visualisierungs-Dashboard**:
  - Backend: FastAPI (`backend/app.py`)
  - Frontend: Streamlit (`frontend/dashboard.py`)
  - Features:
    - 📊 Interaktive Charts (BIP, Preise, Löhne, Profite)
    - 🔍 Firmen-Detailansicht
    - 🏠 Haushalts-Analyse
    - 💾 Daten-Export (JSON/CSV)

- **Training-Script** (`train.py`):
  - PPO Multi-Agent Training
  - TensorFlow Backend (für bessere Windows-Kompatibilität)

### 🚧 Geplant

- Google Cloud Deployment (Training + Hosting)
- Mehr Firmen/Haushalte (skalierbar)
- Erweiterte Marktmechaniken (Kapitalakkumulation, Investitionen)
- Separate Policies pro Firma (echtes MARL statt Parameter Sharing)

## 📂 Projektstruktur

```
VWL-Simulation-RL/
├── env/                      # Environment
│   ├── __init__.py
│   └── economy_env.py        # Volkswirtschafts-Simulation
│
├── backend/                  # API Backend
│   ├── app.py                # FastAPI Server
│   ├── inference.py          # Model Loading & Simulation
│   └── requirements.txt
│
├── frontend/                 # Visualisierungs-Dashboard
│   ├── dashboard.py          # Streamlit App
│   └── requirements.txt
│
├── models/                   # Trainierte Checkpoints (wird erstellt)
│   └── checkpoint_XXXXX/
│
├── train.py                  # Training-Script (PPO)
├── test_env.py               # Environment-Test (ohne Ray)
├── requirements.txt          # Haupt-Dependencies
├── CHANGELOG.md              # Änderungsdokumentation
├── README_DASHBOARD.md       # Dashboard-Dokumentation
│
├── start_backend.bat/.sh     # Backend-Starter
└── start_frontend.bat/.sh    # Frontend-Starter
```

## 🛠️ Installation

### Voraussetzungen

- Python 3.10 oder 3.11
- pip (Python Package Manager)
- Git

### Setup

1. **Repository klonen:**

```bash
git clone https://github.com/H3nri5H/VWL-Simulation-RL.git
cd VWL-Simulation-RL
```

2. **Virtuelle Umgebung erstellen:**

```bash
python -m venv venv

# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

3. **Dependencies installieren:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bibliotheken

**Core:**
- **Ray/RLlib 2.10+**: Multi-Agent RL Framework
- **TensorFlow**: Deep Learning Backend
- **Gymnasium**: Environment API
- **PettingZoo**: Multi-Agent Wrapper

**Visualization:**
- **Streamlit**: Dashboard Framework
- **Plotly**: Interaktive Charts
- **FastAPI**: REST API Backend

**Utils:**
- **NumPy, Pandas**: Datenverarbeitung
- **PyArrow < 21.0.0**: Ray-Kompatibilität

## 🧪 Testing

### Environment testen (ohne Ray/Training)

```bash
python test_env.py
```

**Output:**
```
=== Step 1 ===
GDP: 850.45€
Avg Price: 10.23€
Avg Wage: 8.15€
  firm_0: Price=9.89€, Wage=8.45€, Profit=125.30€, Inventory=95
  firm_1: Price=10.56€, Wage=7.85€, Profit=98.70€, Inventory=102
...
```

### Dashboard testen (ohne trainiertes Modell)

1. Backend starten: `start_backend.bat` (oder `.sh`)
2. Frontend starten: `start_frontend.bat` (oder `.sh`)
3. Dashboard nutzt Random Policy (zufällige Actions)
4. Gut zum Testen der Visualisierungen!

## 📊 Dashboard Nutzung

Siehe **[README_DASHBOARD.md](README_DASHBOARD.md)** für detaillierte Anleitung.

**Kurzübersicht:**

1. **Modell wählen** (oder "random" für Test)
2. **Parameter einstellen** (Firmen, Haushalte, Quartale)
3. **Simulation starten**
4. **Ergebnisse visualisieren**:
   - 📊 Überblick: BIP, Preise, Löhne über Zeit
   - 🏭 Firmen-Details: Einzelne Firma durchleuchten
   - 🏠 Haushalte: Arbeitgeber-Verteilung, Vermögen
   - 💾 Export: JSON/CSV Download

## 🎯 Entwicklungs-Workflow

### 1. Lokal entwickeln (Windows)
```bash
# Code schreiben
# Environment testen
python test_env.py

# Dashboard testen
start_backend.bat & start_frontend.bat
```

### 2. Training (WSL2/Cloud)
```bash
# In WSL2 oder Google Cloud VM:
git pull
python train.py

# Checkpoints landen in ~/ray_results/
```

### 3. Trainierte Modelle visualisieren
```bash
# Checkpoints nach models/ kopieren
cp -r ~/ray_results/PPO_*/checkpoint_* models/

# Dashboard starten
start_backend.bat & start_frontend.bat

# Im Dashboard: Modell auswählen und analysieren
```

## 🚀 Deployment (geplant)

Siehe separate Dokumentation für Google Cloud Deployment:
- Training auf Compute Engine VM
- Checkpoints in Cloud Storage
- Backend auf Cloud Run
- Frontend auf Cloud Run
- Öffentlich zugänglich für Kommilitonen

## 📝 Changelog

Siehe **[CHANGELOG.md](CHANGELOG.md)** für detaillierte Änderungshistorie.

## 👥 Team

DHSH - Fortgeschrittene KI-Anwendungen

## 📄 Lizenz

Studienprojekt - DHSH 2026
