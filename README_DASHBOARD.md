# VWL-Simulation Dashboard

Interaktives Visualisierungs-Dashboard für die Volkswirtschafts-Simulation mit Multi-Agent RL.

## 🚀 Quick Start (Lokal)

### 1. Backend starten

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

Backend läuft auf: http://localhost:8000

### 2. Frontend starten (neues Terminal)

```bash
cd frontend
pip install -r requirements.txt
streamlit run dashboard.py
```

Frontend öffnet sich automatisch im Browser: http://localhost:8501

## 📚 Funktionen

### Backend API (`backend/app.py`)

- **GET /api/models**: Liste aller trainierten Modelle
- **POST /api/simulate**: Simulation starten
- **GET /api/health**: Health Check

### Frontend Dashboard (`frontend/dashboard.py`)

**Setup (Sidebar):**
- Modell-Auswahl (trainierte Checkpoints)
- Parameter: Firmen, Haushalte, Quartale
- Start-Preise und Löhne konfigurieren

**Visualisierungen:**
- 📊 **Überblick**: BIP, Preise, Löhne, Profiteüber Zeit
- 🏭 **Firmen-Details**: Einzelne Firma durchleuchten, durch Quartale scrollen
- 🏠 **Haushalte**: Arbeitgeber-Verteilung, Vermögen, Einkommen
- 💾 **Export**: JSON/CSV Download

## 📂 Projekt-Struktur

```
VWL-Simulation-RL/
├── backend/
│   ├── app.py              # FastAPI Server
│   ├── inference.py        # Simulation Runner
│   └── requirements.txt
│
├── frontend/
│   ├── dashboard.py        # Streamlit App
│   └── requirements.txt
│
├── models/                 # Trainierte Checkpoints
│   └── checkpoint_XXXXX/
│
├── env/
│   └── economy_env.py      # Environment
│
└── test_env.py             # Environment-Test
```

## ⚙️ Konfiguration

### Backend URL ändern

Standardmäßig: `http://localhost:8000`

**Via Umgebungsvariable:**
```bash
export BACKEND_URL="https://your-backend.run.app"
streamlit run dashboard.py
```

**Oder direkt in `dashboard.py`:**
```python
BACKEND_URL = "https://your-backend.run.app"
```

## 📝 Simulation ohne trainiertes Modell

Das Dashboard funktioniert auch **ohne trainierte Modelle**!

- Backend nutzt dann **Random Policy** (zufällige Actions)
- Gut zum Testen der Visualisierungen
- Später: Trainierte Modelle in `models/` Ordner legen

## 🔧 Troubleshooting

### Backend nicht erreichbar

```bash
# Prüfe ob Backend läuft:
curl http://localhost:8000/api/health

# Starte Backend neu:
cd backend
uvicorn app:app --reload
```

### Keine Modelle gefunden

- Erstelle `models/` Ordner im Root
- Oder trainiere erst ein Modell mit `train.py`
- Oder nutze Random Policy (funktioniert auch ohne Modelle)

### Import Errors

```bash
# Backend:
cd backend
pip install -r requirements.txt

# Frontend:
cd frontend
pip install -r requirements.txt
```

## 📚 Weitere Infos

- **Environment**: Siehe `env/economy_env.py`
- **Training**: Siehe `train.py`
- **Testing**: Siehe `test_env.py`
- **Changelog**: Siehe `CHANGELOG.md`

## 🚀 Nächste Schritte

1. **Lokal testen**: Backend + Frontend starten
2. **Modell trainieren**: `train.py` ausführen (auf WSL2/Cloud)
3. **Visualisieren**: Trainierte Modelle im Dashboard analysieren
4. **Deployen**: Backend + Frontend auf Google Cloud (siehe Hauptdoku)
