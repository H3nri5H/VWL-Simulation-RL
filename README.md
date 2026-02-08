# VWL-Simulation-RL

**Multi-Agent Reinforcement Learning Simulation for Economics**

Ein wirtschaftliches Simulationsmodell, in dem KI-gesteuerte Firmen durch Reinforcement Learning lernen, in einem kompetitiven Markt zu agieren. Haushalte kaufen Güter basierend auf Preis, Qualität und Marketing, während Firmen ihre Strategien optimieren müssen, um zu überleben.

---

## 📋 Inhaltsverzeichnis

- [Features](#-features)
- [Installation](#-installation)
- [Schnellstart](#-schnellstart)
- [Projektstruktur](#-projektstruktur)
- [Verwendung](#-verwendung)
- [Sequential Purchasing Modell](#-sequential-purchasing-modell)
- [Konfiguration](#-konfiguration)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Features

### Marktmechanismen
- **Sequential Purchasing**: Haushalte kaufen realistisch von der besten Firma bis Geld/Inventory erschöpft ist
- **Skill-basierter Arbeitsmarkt**: Hochqualifizierte Arbeiter bekommen bessere Jobs
- **Dynamische Preisfindung**: Firmen lernen optimale Preise durch Trial & Error
- **Qualität & Marketing**: Investitionen beeinflussen Kaufentscheidungen

### KI-Training
- **Multi-Agent PPO**: 10 konkurrierende KI-Firmen lernen simultan
- **Erweiterte Action Space**: Preis, Lohn, Marketing, Qualität, Kapazität
- **Reward Shaping**: Marktanteil-Bonus, Inventory-Penalty, Exploration-Penalty
- **Bankruptcy Mechanismus**: Schlechte Strategien führen zu Insolvenz

### Realistische Simulation
- **2000 Haushalte** mit unterschiedlichen Skill-Leveln und Vermögen
- **Supplier Economy**: Arbeitslose arbeiten für Zulieferfirmen
- **25 Features Observation Space**: Umfassende Marktinformationen
- **Reproduzierbare Ergebnisse**: Seed-basierte Simulation

---

## 🚀 Installation

### Voraussetzungen

- **Python 3.9 - 3.11** (empfohlen: 3.10)
- **Git** (um das Repository zu klonen)
- **Mindestens 8 GB RAM** (empfohlen: 16 GB)
- **5 GB freier Festplattenspeicher**

### Schritt 1: Repository klonen

```bash
git clone https://github.com/H3nri5H/VWL-Simulation-RL.git
cd VWL-Simulation-RL
```

### Schritt 2: Python Virtual Environment erstellen

#### Windows (CMD/PowerShell)
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

**Hinweis**: Wenn `venv\Scripts\activate` nicht funktioniert (PowerShell), versuche:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
venv\Scripts\Activate.ps1
```

### Schritt 3: Dependencies installieren

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Wichtig**: Die Installation kann 5-10 Minuten dauern, da PyTorch und Ray große Pakete sind.

### Schritt 4: Installation verifizieren

```bash
python -c "import ray; import torch; print('Installation erfolgreich!')"
```

Wenn keine Fehler erscheinen, ist alles korrekt installiert! ✅

---

## ⚡ Schnellstart

### 1. Training starten

```bash
python train.py
```

**Was passiert:**
- Training läuft für **150 Iterationen** (ca. 3-5 Stunden auf CPU)
- Checkpoints werden alle **10 Iterationen** gespeichert
- Progress wird in der Console angezeigt

**Ausgabe:**
```
======================================================================
  VWL SIMULATION - TRAINING
======================================================================

Fresh training: 150 iterations
Environment: 10 firms, 2000 households
Resources: 4 workers, 0 GPUs

Building algorithm...
Algorithm built

----------------------------------------------------------------------
Iter   Reward       Min        Max        EpLen   
----------------------------------------------------------------------
1      -2.34        -15.23     8.45       100     
2      -1.89        -12.34     9.12       100     
...
```

### 2. Simulation ausführen

```bash
python run_simulation.py
```

**Interaktiver Prozess:**
1. **Checkpoint auswählen** (z.B. Iteration 150)
2. **Seed eingeben** (für Reproduzierbarkeit) oder ENTER für random
3. **Steps festlegen** (Standard: 100)
4. Simulation läuft und speichert CSV-Dateien in `simulation_results/`

**Ergebnis:**
- `firms_checkpoint150_seed12345_20260208_120000.csv`
- `households_checkpoint150_seed12345_20260208_120000.csv`
- `summary_seed12345_20260208_120000.txt`

---

## 📁 Projektstruktur

### Hauptdateien

```
VWL-Simulation-RL/
│
├── train.py                 # Training-Script für KI-Firmen
├── run_simulation.py        # Simulation mit trainiertem Modell
├── config.yaml              # Alle Simulations-Parameter
├── requirements.txt         # Python Dependencies
├── README.md                # Diese Datei
│
├── env/
│   └── economy_env.py       # Gymnasium Environment (Kernlogik)
│
├── checkpoints/             # Gespeicherte Trainings-Checkpoints
│   ├── checkpoint_000010/
│   ├── checkpoint_000020/
│   └── ...
│
├── metrics/                 # Training-Metriken (JSON)
│   ├── iteration_10/
│   └── ...
│
└── simulation_results/      # CSV-Outputs der Simulationen
    ├── firms_*.csv
    ├── households_*.csv
    └── summary_*.txt
```

### Datei-Beschreibungen

#### `train.py` - Training-Script
**Zweck**: Trainiert die KI-Firmen mit PPO (Proximal Policy Optimization)

**Wichtige Funktionen:**
- Lädt Config aus `config.yaml`
- Erstellt PPO Algorithmus mit Ray RLlib
- Trainiert für N Iterationen (Standard: 150)
- Speichert Checkpoints alle 10 Iterationen
- Suppressed verbose Ray-Output für saubere Console

**Verwendung:**
```bash
python train.py              # Fresh training
python train.py --resume     # Resume from latest checkpoint
```

**Output:**
- `checkpoints/checkpoint_NNNNNN/` - Trainierte Modelle
- `metrics/iteration_N/result.json` - Training-Metriken

---

#### `run_simulation.py` - Simulations-Runner
**Zweck**: Führt Simulation mit trainiertem Modell aus und exportiert Daten

**Wichtige Funktionen:**
- Lädt gespeicherte Checkpoints
- Interaktive Checkpoint-Auswahl
- Seed-basierte Reproduzierbarkeit
- Exportiert Daten als CSV (long format für Datenbank-Import)
- Zeigt Initial & Final State an

**Verwendung:**
```bash
python run_simulation.py
```

**Output:**
- CSV-Dateien mit Firmen- und Haushalts-Daten pro Step
- Summary-Text-Datei mit Zusammenfassung
- Konsolen-Output mit wichtigen Statistiken

---

#### `config.yaml` - Konfigurations-Datei
**Zweck**: Zentrale Konfiguration aller Simulations-Parameter

**Sections:**

1. **`environment`**: Umgebungs-Setup
   - `n_firms`: Anzahl Firmen (10)
   - `n_households`: Anzahl Haushalte (2000)
   - `max_steps`: Steps pro Episode (100)

2. **`initial_ranges`**: Start-Werte für Firmen & Haushalte
   - Preis, Lohn, Kapital, Qualität, Marketing
   - Skill-Level, Geld, Vermögensverteilung

3. **`training`**: PPO Training-Parameter
   - Iterations, Learning Rate, Batch Sizes
   - Workers, GPU Settings

4. **`economy`**: Wirtschaftliche Parameter
   - Produktionskosten, Fixkosten
   - Investment-Kosten (Marketing, Qualität, Kapazität)
   - Bankruptcy-Threshold
   - Haushalts-Verhalten (Consumption Rate, Utility Weights)
   - Reward Shaping (Market Share Bonus, Inventory Penalty)

**Wichtig**: Alle Werte in `economy_env.py` werden aus dieser Datei geladen!

---

#### `env/economy_env.py` - Gymnasium Environment
**Zweck**: Kernlogik der Wirtschafts-Simulation

**Klasse**: `SimpleEconomyEnv(MultiAgentEnv)`

**Wichtige Methoden:**

1. **`__init__(config)`**
   - Lädt alle Parameter aus `config.yaml`
   - Initialisiert Observation & Action Space
   - Definiert Adjustment Rates für Actions

2. **`reset(seed)`**
   - Initialisiert Firmen mit Random-Werten aus Config
   - Initialisiert Haushalte mit Skills & Wealth Types
   - Setzt Timestep auf 0
   - Returns: Initial Observations

3. **`step(action_dict)`**
   - **Phase 1**: Firmen nehmen Actions (Preis, Lohn, Marketing, etc.)
   - **Phase 2**: Arbeitsmarkt (Skill-basiertes Matching)
   - **Phase 3**: Produktion (Skill beeinflusst Produktivität)
   - **Phase 4**: Gütermarkt (Sequential Purchasing)
   - **Phase 5**: Profit-Berechnung, Bankruptcy-Check, Rewards
   - Returns: Observations, Rewards, Dones, Infos

4. **`_get_obs(agent_id)`**
   - Erstellt 25-Feature Observation für eine Firma
   - Enthält: Eigene State, Markt-Statistiken, Wettbewerbs-Info

**Action Space**: `MultiDiscrete([5, 5, 3, 2, 3])`
- Price Change: -10%, -5%, 0%, +5%, +10%
- Wage Change: -10%, -5%, 0%, +5%, +10%
- Marketing: Decrease, Keep, Increase
- Quality: No, Yes (Improve)
- Capacity: -1, 0, +1 Employees

**Observation Space**: `Box(25,)` 
- Own State (7): price, wage, employees, inventory, capital, quality, marketing
- Market Stats (6): avg/min/max price/wage
- Aggregates (4): total employed, unemployment, avg household money/skill
- Meta (3): last profit, competitors alive, timestep progress
- Strategic (5): market share, sales trend, inventory ratio, wage/price competitiveness

---

#### `requirements.txt` - Python Dependencies
**Zweck**: Liste aller benötigten Python-Pakete

**Hauptpakete:**
- `ray[rllib]==2.9.0` - Reinforcement Learning Framework
- `torch==2.1.2` - Deep Learning Backend
- `gymnasium==0.29.1` - Environment Interface
- `numpy`, `pandas` - Datenverarbeitung
- `pyyaml` - Config-File Parsing

**Installation:**
```bash
pip install -r requirements.txt
```

---

## 🎮 Sequential Purchasing Modell

### Konzept

Das **Sequential Purchasing** Modell simuliert realistisches Kaufverhalten:

1. **Utility-Berechnung**: Jeder Haushalt berechnet Utility für alle Firmen
   ```
   Utility = (Quality × 0.5 + Marketing × 0.3) / (Price × 1.0)
   ```

2. **Sortierung**: Firmen werden nach Utility sortiert (beste zuerst)

3. **Sequenzieller Kauf**:
   - Haushalt kauft bei **bester Firma** bis:
     - Budget aufgebraucht ODER
     - Inventory der Firma leer
   - Falls Budget übrig: Weiter zur **zweitbesten Firma**
   - Wiederholen bis Budget komplett weg

4. **Random Order**: Haushalts-Reihenfolge wird jeden Step randomisiert (Fairness)

### Beispiel

**Setup:**
- Haushalt Budget: 100€
- Firmen (sortiert nach Utility):
  1. Firma A: 20€/Stk, Inventory: 2, Utility: 0.025
  2. Firma B: 25€/Stk, Inventory: 500, Utility: 0.023
  3. Firma C: 30€/Stk, Inventory: 800, Utility: 0.020

**Ablauf:**
1. Kaufe bei Firma A: 2 Stück × 20€ = 40€ → **Firma A ausverkauft!**
2. Restbudget: 60€
3. Kaufe bei Firma B: 60€ / 25€ = 2.4 → 2 Stück × 25€ = 50€
4. Restbudget: 10€
5. Bei Firma C: 10€ / 30€ = 0.33 → **Zu wenig Geld!**

**Ergebnis:**
- Firma A: Alles verkauft ✅
- Firma B: Etwas verkauft ✅
- Firma C: Nichts verkauft ❌ → Muss Preise senken!

### Vorteile gegenüber Top-3 (50/30/20%)

| Aspekt | Top-3 Split | Sequential |
|--------|-------------|------------|
| Realismus | ❌ Künstlich | ✅ Natürlich |
| Firmen mit Sales | Nur 3 | Alle mit guter Utility |
| Marktdruck | Schwach | Stark auf schwache Firmen |
| Survivors | 3-5 Firmen | 7-9 Firmen |
| Dynamik | Statisch | Sehr dynamisch |

---

## ⚙️ Konfiguration

### Wichtige Parameter anpassen

#### Training verlängern/verkürzen

**In `config.yaml`:**
```yaml
training:
  iterations: 200        # Mehr Iterations = besseres Learning
```

#### Markt vergrößern/verkleinern

```yaml
environment:
  n_firms: 15            # Mehr Firmen = härterer Wettbewerb
  n_households: 3000     # Mehr Haushalte = größerer Markt
```

#### Wirtschaft schwieriger machen

```yaml
economy:
  production:
    fixed_costs: 150.0   # Höhere Fixkosten
  bankruptcy:
    threshold: -1000.0   # Schnellerer Bankrott
```

#### Haushalts-Verhalten ändern

```yaml
economy:
  households:
    consumption_rate: 0.8              # Mehr Konsum (80% statt 70%)
    utility_price_weight: 1.5          # Preis wichtiger
    utility_quality_weight: 0.8        # Qualität wichtiger
```

### GPU Training aktivieren

**In `config.yaml`:**
```yaml
training:
  resources:
    num_gpus: 1          # 1 GPU verwenden (falls vorhanden)
```

**Hinweis**: Erfordert CUDA-fähige GPU und `torch` mit CUDA-Support.

---

## 🔧 Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'ray'`

**Lösung:**
```bash
pip install -r requirements.txt
```

---

### Problem: Training sehr langsam

**Mögliche Ursachen:**
1. Zu wenig RAM → Close other applications
2. CPU zu schwach → Reduce `num_env_runners` in `config.yaml`:
   ```yaml
   training:
     resources:
       num_env_runners: 2  # Statt 4
   ```
3. Zu viele Haushalte → Reduce in `config.yaml`:
   ```yaml
   environment:
     n_households: 1000    # Statt 2000
   ```

---

### Problem: `RuntimeError: CUDA out of memory`

**Lösung**: GPU Training deaktivieren
```yaml
training:
  resources:
    num_gpus: 0
```

---

### Problem: Checkpoints werden nicht gespeichert

**Check**:
1. Schreibrechte im Projektordner?
2. Genug Speicherplatz? (ca. 500 MB pro Checkpoint)

**Lösung**:
```bash
# Windows
icacls checkpoints /grant %USERNAME%:F

# Linux/macOS
chmod -R 755 checkpoints/
```

---

### Problem: Alle Firmen gehen bankrott

**Mögliche Ursachen:**
1. Zu harsche Bankruptcy-Threshold
2. Fixed Costs zu hoch
3. Model noch nicht trainiert (frühe Iterationen)

**Lösungen**:
1. **Sanftere Bankruptcy**:
   ```yaml
   economy:
     bankruptcy:
       threshold: -5000.0  # Mehr Spielraum
   ```

2. **Niedrigere Fixkosten**:
   ```yaml
   economy:
     production:
       fixed_costs: 50.0   # Statt 100
   ```

3. **Mehr Training**: Warte bis Iteration 50+

---

### Problem: `python: command not found`

**Windows**: Verwende `py` statt `python`
```bash
py train.py
```

**macOS/Linux**: Verwende `python3`
```bash
python3 train.py
```

---

## 📊 Erwartete Ergebnisse

### Nach 50 Iterationen (Early Learning)
- **Survivors**: 5-7 Firmen
- **Avg Reward**: -5 bis 0
- **Bankruptcy Rate**: 30-50%
- **Market Share**: Ungleich verteilt

### Nach 100 Iterationen (Intermediate)
- **Survivors**: 6-8 Firmen
- **Avg Reward**: 0 bis +5
- **Bankruptcy Rate**: 20-30%
- **Market Share**: Gleichmäßiger

### Nach 150 Iterationen (Converged)
- **Survivors**: 7-9 Firmen ✅
- **Avg Reward**: +3 bis +8
- **Bankruptcy Rate**: 10-20%
- **Market Share**: 8-15% pro Firma
- **Dynamik**: Stabile aber wechselnde Marktanteile

---

## 📚 Weitere Informationen

### Verwendete Technologien
- **Ray RLlib**: Multi-Agent Reinforcement Learning Framework
- **PyTorch**: Deep Learning Backend
- **Gymnasium**: Standard-Interface für RL Environments
- **PPO**: Proximal Policy Optimization Algorithmus

### Nützliche Ressourcen
- [Ray RLlib Dokumentation](https://docs.ray.io/en/latest/rllib/index.html)
- [Gymnasium Dokumentation](https://gymnasium.farama.org/)
- [PPO Paper (Schulman et al.)](https://arxiv.org/abs/1707.06347)

---

## 📝 Lizenz

Dieses Projekt ist für akademische Zwecke entwickelt.

---

## 👥 Kontakt

Bei Fragen oder Problemen, bitte ein Issue auf GitHub erstellen.

---

**Viel Erfolg beim Training! 🚀**
