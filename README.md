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
- [Sequential Purchasing mit Preissensitivität](#-sequential-purchasing-mit-preissensitivit%C3%A4t)
- [Konfiguration](#-konfiguration)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Features

### Marktmechanismen
- **Sequential Purchasing**: Haushalte kaufen realistisch von der besten Firma bis Geld/Inventory erschöpft ist
- **🆕 Preissensitive Haushalte**: Jeder Haushalt hat max_acceptable_price - ignoriert zu teure Firmen komplett
- **Skill-basierter Arbeitsmarkt**: Hochqualifizierte Arbeiter bekommen bessere Jobs
- **Dynamische Preisfindung**: Firmen lernen optimale Preise durch Trial & Error
- **Qualität & Marketing**: Investitionen beeinflussen Kaufentscheidungen

### KI-Training
- **Multi-Agent PPO**: 10 konkurrierende KI-Firmen lernen simultan
- **Erweiterte Action Space**: Preis, Lohn, Marketing, Qualität, Kapazität
- **Rebalanced Rewards**: Profit weniger dominant (scale: 1000), Bankruptcy schwer bestraft (-50,000)
- **🆕 Survivor Diversity Penalty**: KI wird bestraft wenn zu viele Firmen bankrott gehen
- **Hard Employee Cap**: Verhindert Monopole (max 150 Mitarbeiter)

### Realistische Simulation
- **3000 Haushalte** (erhöht von 2000) mit unterschiedlichen Skill-Leveln, Vermögen und Preislimits
- **Kleinere Firmen** (80-120 statt 150-250 Mitarbeiter) für bessere Konkurrenz
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
- Training läuft für **200 Iterationen** (ca. 4-6 Stunden auf CPU)
- Checkpoints werden alle **10 Iterationen** gespeichert
- Progress wird in der Console angezeigt

**Ausgabe:**
```
======================================================================
  VWL SIMULATION - TRAINING
======================================================================

Fresh training: 200 iterations
Environment: 10 firms, 3000 households
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
1. **Checkpoint auswählen** (z.B. Iteration 200)
2. **Seed eingeben** (für Reproduzierbarkeit) oder ENTER für random
3. **Steps festlegen** (Standard: 100)
4. Simulation läuft und speichert CSV-Dateien in `simulation_results/`

**Ergebnis:**
- `firms_checkpoint200_seed12345_20260208_120000.csv`
- `households_checkpoint200_seed12345_20260208_120000.csv`
- `summary_seed12345_20260208_120000.txt`

---

## 🎮 Sequential Purchasing mit Preissensitivität

### Konzept

Das **Sequential Purchasing** Modell mit **Preissensitivität** simuliert hochrealistisches Kaufverhalten:

#### Phase 1: Preisfilterung (NEU! 🆕)
Jeder Haushalt hat `max_acceptable_price` (10-100€):
- **Low-Budget Haushalt** (max: 30€): Kauft nur bei günstigen Firmen
- **Medium Haushalt** (max: 60€): Mittleres Preissegment
- **Premium Haushalt** (max: 100€): Akzeptiert auch teure Preise

**Firmen über dem Limit werden KOMPLETT ignoriert!**

#### Phase 2: Utility-Berechnung
Für alle **erschwinglichen** Firmen:
```
Utility = (Quality × 0.5 + Marketing × 0.3) / (Price × 1.0)
```

#### Phase 3: Sequential Purchasing
1. **Sortierung**: Firmen nach Utility (beste zuerst)
2. **Sequenzieller Kauf**:
   - Haushalt kauft bei **bester Firma** bis:
     - Budget aufgebraucht ODER
     - Inventory der Firma leer
   - Falls Budget übrig: Weiter zur **zweitbesten Firma**
   - Wiederholen bis Budget komplett weg
3. **Random Order**: Haushalts-Reihenfolge jeden Step randomisiert (Fairness)

### Beispiel

**Setup:**
- Haushalt: Budget 100€, max_acceptable_price: 40€
- Firmen:
  - Firma A: 25€/Stk, Quality: 0.7, Marketing: 0.5, Inventory: 3
  - Firma B: 35€/Stk, Quality: 0.8, Marketing: 0.6, Inventory: 500  
  - Firma C: 50€/Stk, Quality: 0.9, Marketing: 0.7, Inventory: 800 (ZU TEUER!)

**Ablauf:**
1. **Preisfilter**: Firma C wird ignoriert (50€ > 40€)
2. **Utility-Berechnung**:
   - Firma A: (0.7×0.5 + 0.5×0.3) / 25 = 0.020
   - Firma B: (0.8×0.5 + 0.6×0.3) / 35 = 0.016
3. **Kaufen bei Firma A**: 3 Stück × 25€ = 75€ → **Ausverkauft!**
4. **Restbudget**: 25€
5. **Zu wenig für Firma B** (25€ < 35€)

**Ergebnis:**
- Firma A: Alles verkauft ✅
- Firma B: Nichts verkauft (zu teuer für Budget)
- Firma C: Ignoriert (über Preislimit)

### Markt-Stratifizierung

Durch Preissensitivität entstehen **natürliche Marktsegmente**:

| Segment | Preis | Zielgruppe | Strategie |
|---------|-------|------------|----------|
| Budget | 10-30€ | Low-wealth HH (30%) | Volumen über Preis |
| Mittelklasse | 30-60€ | Medium-wealth HH (50%) | Balance Preis/Qualität |
| Premium | 60-100€ | High-wealth HH (20%) | Qualität über Preis |

**KI lernt:**
- Nicht alle Firmen müssen billig sein!
- Premium-Strategie kann profitabel sein
- Marktsegmentierung verhindert "Race to the Bottom"

---

## ⚙️ Konfiguration

### Wichtige Parameter anpassen

#### Training verlängern/verkürzen

**In `config.yaml`:**
```yaml
training:
  iterations: 250        # Mehr Iterations = besseres Learning
```

#### Markt vergrößern/verkleinern

```yaml
environment:
  n_firms: 15            # Mehr Firmen = härterer Wettbewerb
  n_households: 5000     # Mehr Haushalte = größerer Markt
```

#### Wirtschaft schwieriger machen

```yaml
economy:
  production:
    fixed_costs: 200.0   # Höhere Fixkosten
  bankruptcy:
    threshold: -1000.0   # Schnellerer Bankrott
    penalty_reward: -100000.0  # Noch härtere Strafe
```

#### Haushalts-Preissensitivität ändern

```yaml
initial_ranges:
  households:
    max_acceptable_price:
      min: 20.0          # Alle müssen mindestens 20€ akzeptieren
      max: 80.0          # Niemand zahlt über 80€
```

#### Firmen-Größe limitieren

```yaml
economy:
  max_employees_hard_cap: 100   # Kleinere Monopole
  
initial_ranges:
  firms:
    max_employees:
      min: 50
      max: 80
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

### Problem: Fast alle Firmen gehen sofort bankrott

**Ursache**: Neue Bankruptcy Penalty (-50,000) ist extrem hoch, Model muss erst lernen

**Lösungen:**

1. **Mehr Training**: Warte bis Iteration 100+ (frühe Phasen haben viele Bankruptcies)

2. **Sanftere Penalty** (für Experimente):
   ```yaml
   economy:
     bankruptcy:
       penalty_reward: -10000.0  # Statt -50000
   ```

3. **Höheres Initial Capital**:
   ```yaml
   initial_ranges:
     firms:
       capital:
         min: 10000.0   # Statt 5000
         max: 20000.0   # Statt 10000
   ```

4. **Niedrigere Fixkosten**:
   ```yaml
   economy:
     production:
       fixed_costs: 100.0  # Statt 150
   ```

---

### Problem: Nur 3 Firmen überleben (Monopole)

**Ursache**: Model hat falsche Strategie gelernt (alte Checkpoints vor Rebalancing)

**Lösung**: **Fresh Training** mit neuen Parametern!

```bash
# Alte Checkpoints löschen
rm -rf checkpoints/*  # Linux/Mac
del /s /q checkpoints\*  # Windows

# Neu trainieren
python train.py
```

---

### Problem: Haushalte kaufen nichts (Revenue = 0)

**Ursache**: Alle Firmen sind zu teuer für Haushalts-Preislimits

**Check**:
```python
# In run_simulation.py Output schauen:
Avg Firm Price: 85€
Avg HH Max Price: 55€  # ← Problem!
```

**Lösung**: Preis-Range anpassen
```yaml
initial_ranges:
  firms:
    price:
      min: 15.0    # Niedriger starten
      max: 45.0    # Nicht zu hoch
```

---

### Problem: Training sehr langsam (mit 3000 Haushalten)

**Mögliche Ursachen:**
1. Zu wenig RAM → Close other applications
2. CPU zu schwach → Reduce workers:
   ```yaml
   training:
     resources:
       num_env_runners: 2  # Statt 4
   ```
3. Zu viele Haushalte → Reduce (aber mindestens 2000!):
   ```yaml
   environment:
     n_households: 2000    # Statt 3000
   ```

**Hinweis**: Mit 3000 HH ist Training ca. 30% langsamer als mit 2000 HH, aber Ergebnisse sind besser!

---

### Problem: `ModuleNotFoundError: No module named 'ray'`

**Lösung:**
```bash
pip install -r requirements.txt
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

### Problem: Survivor Diversity Penalty zu hoch

**Symptom**: Rewards am Ende plötzlich massiv negativ

**Check in `config.yaml`:**
```yaml
economy:
  reward:
    survivor_diversity_threshold: 5     # Penalty wenn < 5 survivors
    survivor_diversity_penalty: 10000   # Pro fehlender Firma
```

**Beispiel**: 3 Survivors → (5-3) × 10,000 = -20,000 Penalty **pro Survivor**!

**Lösung**: Penalty reduzieren oder Threshold senken
```yaml
economy:
  reward:
    survivor_diversity_threshold: 4     # Weniger streng
    survivor_diversity_penalty: 5000    # Sanfter
```

---

## 📊 Erwartete Ergebnisse (Nach Rebalancing)

### Nach 50 Iterationen (Early Learning)
- **Survivors**: 3-5 Firmen ⚠️ (noch viele Bankruptcies)
- **Avg Reward**: -500 bis -100 (Bankruptcy Penalties dominieren)
- **Bankruptcy Rate**: 50-70%
- **Market Share**: Stark ungleich
- **Avg Firm Size**: 60-90 Mitarbeiter

### Nach 100 Iterationen (Intermediate)
- **Survivors**: 5-7 Firmen ✅
- **Avg Reward**: -50 bis +50
- **Bankruptcy Rate**: 30-50%
- **Market Share**: Gleichmäßiger werdend
- **Avg Firm Size**: 70-100 Mitarbeiter
- **Price Stratification**: Beginnt sich zu zeigen

### Nach 150 Iterationen (Good)
- **Survivors**: 6-8 Firmen ✅✅
- **Avg Reward**: +20 bis +80
- **Bankruptcy Rate**: 20-30%
- **Market Share**: 10-16% pro Firma
- **Avg Firm Size**: 80-110 Mitarbeiter
- **Price Range**: 20-70€ (Budget bis Premium)

### Nach 200 Iterationen (Converged) ⭐
- **Survivors**: 7-9 Firmen ✅✅✅
- **Avg Reward**: +40 bis +100
- **Bankruptcy Rate**: 10-20%
- **Market Share**: 10-14% pro Firma (sehr balanced!)
- **Avg Firm Size**: 85-115 Mitarbeiter
- **Price Stratification**: Klar erkennbar:
  - Budget: 2-3 Firmen bei 15-35€
  - Mittelklasse: 3-4 Firmen bei 35-60€
  - Premium: 1-2 Firmen bei 60-85€
- **Employment Rate**: 75-85%
- **Market Dynamics**: Stabil aber mit wechselnden Marktanteilen

### Vergleich: Alte vs. Neue Parameter

| Metrik | ALT (2000 HH, kleine Penalty) | NEU (3000 HH, große Penalty) |
|--------|-------------------------------|------------------------------|
| Survivors | 3-5 ❌ | 7-9 ✅ |
| Monopole | Häufig | Selten |
| Avg Firm Size | 150-220 | 80-115 |
| Bankruptcy Learning | Schwach | Stark |
| Market Balance | Ungleich | Sehr gleichmäßig |
| Price Competition | Niedrig | Hoch |
| Reward Stability | Volatil | Stabiler |

---

## 📚 Weitere Informationen

### Verwendete Technologien
- **Ray RLlib**: Multi-Agent Reinforcement Learning Framework
- **PyTorch**: Deep Learning Backend
- **Gymnasium**: Standard-Interface für RL Environments
- **PPO**: Proximal Policy Optimization Algorithmus

### Key Features dieser Implementation
1. **Price-Sensitive Sequential Purchasing**: Einzigartige Kombination von Preislimits + sequenziellem Kauf
2. **Rebalanced Rewards**: Bankruptcy-Vermeidung wichtiger als kurzfristige Profite
3. **Survivor Diversity Incentive**: KI lernt, Konkurrenz am Leben zu halten
4. **Hard Capacity Caps**: Verhindert Monopol-Bildung
5. **Skill-Based Matching**: Realistische Arbeitsmarkt-Dynamik

### Nützliche Ressourcen
- [Ray RLlib Dokumentation](https://docs.ray.io/en/latest/rllib/index.html)
- [Gymnasium Dokumentation](https://gymnasium.farama.org/)
- [PPO Paper (Schulman et al.)](https://arxiv.org/abs/1707.06347)

---

## 🆕 Changelog

### Version 2.0 (Feb 2026) - Major Rebalancing

**Neue Features:**
- ✅ Price-sensitive households (max_acceptable_price)
- ✅ Hard employee cap (150 max)
- ✅ Survivor diversity penalty
- ✅ Rebalanced rewards (profit scale 100 → 1000)

**Parameter Changes:**
- Households: 2000 → 3000
- Max employees: 150-250 → 80-120
- Bankruptcy penalty: -20 → -50,000
- Fixed costs: 100 → 150
- Training iterations: 150 → 200
- Quality/Marketing ranges: Narrowed for fairness

**Expected Impact:**
- More survivors (7-9 vs 3-5)
- Better market balance
- Natural price stratification
- Stronger bankruptcy avoidance learning

---

## 📝 Lizenz

Dieses Projekt ist für akademische Zwecke entwickelt.

---

## 👥 Kontakt

Bei Fragen oder Problemen, bitte ein Issue auf GitHub erstellen.

---

**Viel Erfolg beim Training! 🚀**

**WICHTIG**: Für beste Ergebnisse mit neuen Parametern **fresh training** starten (alte Checkpoints löschen)!
