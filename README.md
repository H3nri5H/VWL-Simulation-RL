# VWL-Simulation-RL

**Multi-Agent Reinforcement Learning Simulation for Economics**

Ein wirtschaftliches Simulationsmodell, in dem KI-gesteuerte Firmen durch Reinforcement Learning lernen, in einem kompetitiven Markt zu agieren. Haushalte kaufen Güter basierend auf **Preis, Qualität, Marketing UND ihrem Einkommen**, während Firmen ihre Strategien optimieren müssen, um zu überleben.

---

## 📋 Inhaltsverzeichnis

- [Features](#-features)
- [Installation](#-installation)
- [Schnellstart](#-schnellstart)
- [Projektstruktur](#-projektstruktur)
- [Verwendung](#-verwendung)
- [HTTP API](#-http-api)
- [Einkommensbasiertes Kaufverhalten](#-einkommensbasiertes-kaufverhalten)
- [Konfiguration](#-konfiguration)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Features

### Marktmechanismen

- **🆕 Einkommensbasierte Präferenzen**: Reiche kaufen Qualität, Arme kaufen billig - verhindert "Race to the Bottom"!
- **Sequential Purchasing**: Haushalte kaufen realistisch von der besten Firma bis Geld/Inventory erschöpft ist
- **Preissensitive Haushalte**: Jeder Haushalt hat max_acceptable_price - ignoriert zu teure Firmen komplett
- **Unbegrenzte Qualität**: Firmen können Quality & Marketing bis 2.0 steigern (statt 1.0 Cap)
- **Skill-basierter Arbeitsmarkt**: Hochqualifizierte Arbeiter bekommen bessere Jobs
- **Dynamische Preisfindung**: Firmen lernen optimale Preise durch Trial & Error

### KI-Training

- **Multi-Agent PPO**: 10 konkurrierende KI-Firmen lernen simultan
- **Erweiterte Action Space**: Preis, Lohn, Marketing, Qualität, Kapazität
- **Rebalanced Rewards**: Profit weniger dominant (scale: 1000), Bankruptcy schwer bestraft (-50,000)
- **Survivor Diversity Penalty**: KI wird bestraft wenn zu viele Firmen bankrott gehen
- **Hard Employee Cap**: Verhindert Monopole (max 150 Mitarbeiter)

### Realistische Simulation

- **3000 Haushalte** mit unterschiedlichen:
  - Skill-Leveln (0.3-1.0)
  - Vermögen (100-200€ Start)
  - Preislimits (10-100€)
  - **Einkommensklassen** (30% arm, 50% mittel, 20% reich)
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

## 💵 Einkommensbasiertes Kaufverhalten

### Das Problem: "Race to the Bottom"

**VORHER** (nur Preis zählt):

```
Utility = (Quality × 0.5 + Marketing × 0.3) / Price

Billigste Firma IMMER besser:
  Firma A: 10€, Quality 0.7  → Utility = 0.085  ← GEWINNT IMMER
  Firma B: 50€, Quality 1.5  → Utility = 0.039  ← Verliert

Ergebnis: Alle Firmen bei 10€ Minimum-Preis! ❌
```

### Die Lösung: Einkommensbasierte Präferenzen 🆕

**JETZT** (Einkommen bestimmt Präferenzen):

Jeder Haushalt hat einen **Wealth Type** (arm/mittel/reich) mit unterschiedlichen **Utility Weights**:

#### **Arme Haushalte** (30% der Bevölkerung)

```python
Utility = (Quality × 0.3 + Marketing × 0.21) / (Price × 1.5)
                ↑ wenig wichtig               ↑ SEHR wichtig!

Beispiel:
  Budget-Firma:  10€, Quality 0.7  → Utility = 0.0315  ← BEST!
  Premium-Firma: 70€, Quality 1.8  → Utility = 0.0081  ← Zu teuer

Verhalten: Kaufen NUR billige Produkte!
```

#### **Mittelschicht** (50% der Bevölkerung)

```python
Utility = (Quality × 0.5 + Marketing × 0.3) / (Price × 1.0)
                ↑ wichtig                ↑ normal

Beispiel:
  Budget-Firma:  10€, Quality 0.7  → Utility = 0.065
  Mittel-Firma:  40€, Quality 1.2  → Utility = 0.0225  ← Gutes Verhältnis
  Premium-Firma: 70€, Quality 1.8  → Utility = 0.0194

Verhalten: Balance zwischen Preis und Qualität!
```

#### **Reiche Haushalte** (20% der Bevölkerung)

```python
Utility = (Quality × 0.9 + Marketing × 0.45) / (Price × 0.5)
                ↑ SEHR wichtig!            ↑ weniger wichtig

Beispiel:
  Budget-Firma:  10€, Quality 0.7  → Utility = 0.171
  Premium-Firma: 70€, Quality 1.8  → Utility = 0.0656  ← Qualität lohnt sich!

Verhalten: Zahlen GERNE mehr für hohe Qualität!
```

---

### Natürliche Marktsegmentierung

Durch einkommensbasierte Präferenzen entstehen **drei profitable Segmente**:

| Segment        | Preis   | Quality | Zielgruppe                  | Strategie                     |
| -------------- | ------- | ------- | --------------------------- | ----------------------------- |
| **Budget**     | 15-35€  | 0.7-1.0 | Arme (30%) + Mittel (50%)   | Volumen durch niedrigen Preis |
| **Mainstream** | 35-60€  | 1.0-1.5 | Mittel (50%) + Reiche (20%) | Balance Preis/Qualität        |
| **Premium**    | 60-100€ | 1.5-2.0 | Reiche (20%)                | Hohe Marge durch Qualität     |

**Erwartete Firma-Verteilung nach Training:**

- 3-4 Budget-Firmen (billiger Preis, okay Quality)
- 3-4 Mainstream-Firmen (mittlerer Preis, gute Quality)
- 2-3 Premium-Firmen (hoher Preis, exzellente Quality)

**ALLE können profitabel sein!** ✅

---

### Beispiel: Zwei profitable Strategien

#### **Budget-Firma** (Firma A)

```
Preis: 20€
Quality: 0.8
Marketing: 0.5
Employees: 120

Kunden:
  - Arme HH (900): Utility = 0.031  ← SEHR GUT für diese Gruppe
  - Mittel HH (1500): Utility = 0.055  ← Okay
  - Reiche HH (600): Utility = 0.110  ← Auch für Reiche akzeptabel!

Total: ~2400 potenzielle Kunden (80%!)
Umsatz: 2400 × 20€ = 48,000€ per Step
Profit: ~25,000€ per Step  ✅
```

#### **Premium-Firma** (Firma B)

```
Preis: 75€
Quality: 1.8
Marketing: 1.5
Employees: 90

Kunden:
  - Arme HH (900): Utility = 0.0081  ← ZU TEUER, kaufen nicht
  - Mittel HH (1500): Utility = 0.030  ← Nur top earners kaufen
  - Reiche HH (600): Utility = 0.066  ← PERFEKT für diese Gruppe!

Total: ~800 potenzielle Kunden (27%)
Umsatz: 800 × 75€ = 60,000€ per Step  ← MEHR als Budget!
Profit: ~30,000€ per Step  ✅✅

Warum profitabler?
  - Höhere Marge (75€ vs 20€)
  - Weniger Competition um reiche Kunden
  - Kleinere Firma = niedrigere Lohnkosten
```

**Beide Strategien funktionieren!** Kein "Race to the Bottom" mehr! 🎉

---

## 💪 Unbegrenzte Qualität & Marketing

### Vorher: Artificial Cap bei 1.0

```yaml
OLD:
  Quality: max 1.0  ❌
  Marketing: max 1.0  ❌

Problem: Premium-Firmen können sich nicht differenzieren!
```

### Jetzt: Cap bei 2.0

```yaml
NEW:
  Quality: max 2.0  ✅ (kann unbegrenzt investieren!)
  Marketing: max 2.0  ✅ (Premium-Branding möglich!)

Kosten:
  Quality +0.05: 50€ (erhöht von 30€)
  Marketing +0.1: 20€

Beispiel Premium-Entwicklung:
  Start: Quality 0.7, Marketing 0.5
  Step 20: Quality 1.2, Marketing 1.0  (investiert 10,000€)
  Step 50: Quality 1.8, Marketing 1.5  (investiert weitere 15,000€)

Ergebnis: Echte Premium-Marke mit Alleinstellungsmerkmal!
```

**Premium-Firmen können jetzt richtig in Qualität investieren!** 🚀

---

## 🌐 HTTP API

Du kannst Training und Simulation per HTTP starten.

### API starten

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### API + DB-Services mit Docker starten

```bash
docker compose up -d --build
```

Nur API im Container starten:

```bash
docker compose up -d --build app
```

Wichtig für den Uploader im Docker-Stack:

1. In `database/.env` entweder `DATABASE_URL` setzen oder `DB_USER`, `DB_PASSWORD`, `DB_NAME` ergänzen.
2. Für `cloud-sql-proxy` müssen Google Application Default Credentials verfügbar sein (z. B. lokal: `gcloud auth application-default login`).

### Training starten

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"resume": false}'
```

### Simulation starten

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "steps": 365}'
```

Wenn kein `checkpoint` angegeben wird, nutzt die API automatisch den neuesten Checkpoint.

Status prüfen (mit `pid` aus der Start-Antwort):

```bash
curl http://localhost:8000/simulation-status/<pid>
```

Optional synchron warten bis Simulation fertig ist:

```bash
curl -X POST "http://localhost:8000/simulate?wait_for_finish=true" \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "steps": 365}'
```

Optional mit Checkpoint:

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{"checkpoint": "./checkpoints/checkpoint_000700", "seed": 42, "steps": 365}'
```

### Checkpoints für Frontend-Dropdown laden

```bash
curl http://localhost:8000/checkpoints
```

Die Antwort enthält `path` und `iteration`.

Hinweis: `POST /train` und `POST /simulate` starten jeweils einen Hintergrundprozess und geben `pid` + `log_file` zurück.

Training stoppen:

```bash
curl -X POST "http://localhost:8000/stop-training"
```

---

## ⚙️ Konfiguration

### Einkommens-Präferenzen anpassen

**In `config.yaml`:**

```yaml
economy:
  households:
    # Wie stark beeinflussen Preis/Qualität/Marketing die Kaufentscheidung?
    wealth_utility_modifiers:
      low: # Arme Haushalte
        price_weight: 1.5 # SEHR preis-sensitiv
        quality_weight: 0.6 # Quality weniger wichtig
        marketing_weight: 0.7 # Ads weniger wirksam

      medium: # Mittelschicht
        price_weight: 1.0 # Normal
        quality_weight: 1.0 # Normal
        marketing_weight: 1.0 # Normal

      high: # Reiche
        price_weight: 0.5 # Preis weniger wichtig!
        quality_weight: 1.8 # Quality SEHR wichtig!
        marketing_weight: 1.5 # Branding wirkt stark
```

**Beispiel: Noch extremere Segmentierung**

```yaml
low:
  price_weight: 2.0 # Arme kaufen NUR billig
  quality_weight: 0.3 # Quality fast egal

high:
  price_weight: 0.3 # Reiche ignorieren Preis
  quality_weight: 2.5 # Nur beste Quality zählt
```

### Quality/Marketing Caps anpassen

```yaml
economy:
  quality_bounds:
    min: 0.1
    max: 3.0 # Noch höhere Qualität möglich!

  marketing_bounds:
    min: 0.1
    max: 3.0 # Luxury-Branding!

  investment_costs:
    quality_improvement: 100.0 # Teurer machen (war 50)
    marketing_per_level: 50.0 # Teurer machen (war 20)
```

### Markt vergrößern/verkleinern

```yaml
environment:
  n_firms: 15 # Mehr Firmen = härterer Wettbewerb
  n_households: 5000 # Mehr Haushalte = größerer Markt
```

---

## 🔧 Troubleshooting

### Problem: Immer noch alle Firmen bei 10€

**Ursache**: Training zu kurz oder alte Checkpoints

**Lösung**:

1. **Fresh Training** starten (alte Checkpoints löschen)
2. Mindestens bis **Iteration 100** warten
3. Check nach Iteration 100:
   - Sollte 3-4 Firmen über 30€ geben
   - Min 2 Firmen über 50€

**Falls immer noch Problem nach Iter 100:**

```yaml
# Reiche noch preisunsensibler machen
high:
  price_weight: 0.3 # Statt 0.5
  quality_weight: 2.0 # Statt 1.8
```

---

### Problem: Nur 2 Firmen verkaufen (Market Concentration)

**Ursache**: Quality/Marketing zu ähnlich bei allen Firmen

**Lösung**: Größere Initial Ranges

```yaml
initial_ranges:
  firms:
    quality:
      min: 0.5 # Größere Spreizung
      max: 0.9 # (war 0.65-0.75)
    marketing:
      min: 0.3
      max: 0.7 # (war 0.45-0.55)
```

---

### Problem: Premium-Firmen gehen alle bankrott

**Ursache**: Zu wenig reiche Haushalte

**Lösung**: Mehr Reiche

```yaml
initial_ranges:
  households:
    wealth_distribution:
      low: 0.2 # Weniger Arme (war 0.3)
      medium: 0.5 # Gleich
      high: 0.3 # Mehr Reiche! (war 0.2)
```

---

### Problem: Fast alle Firmen gehen bankrott (Iteration 1-50)

**Das ist NORMAL!** Siehe Hauptdokumentation für Details.

**Kurz**: Bankruptcy Penalty (-50,000) ist sehr hoch - KI muss erst lernen zu überleben.

**Erwartung**:

- Iteration 20: 1-2 Bankruptcies, 8-9 Survivors ✅
- Iteration 50: 0-1 Bankruptcies, 9-10 Survivors ✅
- Iteration 100+: Fast keine Bankruptcies ✅

---

## 📊 Erwartete Ergebnisse (Mit neuen Features)

### Nach 100 Iterationen

- **Survivors**: 8-9 Firmen ✅
- **Price Range**: 15-65€ (Stratification beginnt!)
- **Quality Range**: 0.7-1.3
- **Bankruptcy Rate**: 10-20%
- **Firmengröße**: 70-110 Mitarbeiter

**Segment-Verteilung:**

- Budget (<30€): 3-4 Firmen
- Mainstream (30-60€): 3-4 Firmen
- Premium (>60€): 1-2 Firmen

### Nach 200 Iterationen (Optimal) ⭐

- **Survivors**: 8-10 Firmen ✅✅✅
- **Price Range**: 15-80€ (KLARE Stratification!)
- **Quality Range**: 0.7-1.8 (Premium deutlich höher)
- **Bankruptcy Rate**: <10%
- **Firmengröße**: 80-120 Mitarbeiter
- **Market Share**: 8-13% pro Firma (sehr balanced!)

**Segment-Verteilung:**

- Budget (15-30€): 3-4 Firmen, Quality 0.7-1.0
- Mainstream (30-60€): 3-4 Firmen, Quality 1.0-1.4
- Premium (60-85€): 2-3 Firmen, Quality 1.5-1.8

**Profitabilität:**

- Budget-Firmen: +15k-25k per Episode (Volumen)
- Mainstream: +20k-30k per Episode (Balance)
- Premium: +25k-35k per Episode (Marge!) ← BESTE!

---

## 📚 Weitere Informationen

### Verwendete Technologien

- **Ray RLlib**: Multi-Agent Reinforcement Learning Framework
- **PyTorch**: Deep Learning Backend
- **Gymnasium**: Standard-Interface für RL Environments
- **PPO**: Proximal Policy Optimization Algorithmus

### Key Features dieser Implementation

1. **🆕 Einkommensbasierte Utility**: Einzigartig! Reiche kaufen Quality, Arme kaufen billig
2. **Unbegrenzte Qualität**: Premium-Firmen können sich wirklich differenzieren (bis 2.0)
3. **Price-Sensitive Sequential Purchasing**: Realistische Kaufreihenfolge mit Preislimits
4. **Rebalanced Rewards**: Bankruptcy-Vermeidung wichtiger als kurzfristige Profite
5. **Survivor Diversity Incentive**: KI lernt, Konkurrenz am Leben zu halten
6. **Hard Capacity Caps**: Verhindert Monopol-Bildung
7. **Skill-Based Matching**: Realistische Arbeitsmarkt-Dynamik

### Nützliche Ressourcen

- [Ray RLlib Dokumentation](https://docs.ray.io/en/latest/rllib/index.html)
- [Gymnasium Dokumentation](https://gymnasium.farama.org/)
- [PPO Paper (Schulman et al.)](https://arxiv.org/abs/1707.06347)

---

## 🆕 Changelog

### Version 2.1 (Feb 2026) - Einkommensbasierte Präferenzen

**MAJOR UPDATE:**

- ✅ **Wealth-based utility preferences** (rich prefer quality, poor prefer price)
- ✅ **Unbegrenzte Quality/Marketing** (1.0 → 2.0 cap)
- ✅ **Natural market segmentation** (budget/mainstream/premium all viable)
- ✅ **Quality investment cost increased** (30 → 50€ for balance)

**Löst:**

- ❌ "Race to the Bottom" (alle Firmen bei 10€)
- ❌ Market Concentration (nur 2 Firmen verkaufen)
- ❌ Artificial quality cap (Premium-Firmen können nicht differenzieren)

**Expected Impact:**

- 3-4 Budget-Firmen (15-30€)
- 3-4 Mainstream-Firmen (30-60€)
- 2-3 Premium-Firmen (60-85€)
- ALLE profitabel! ✅

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

---

## 📝 Lizenz

Dieses Projekt ist für akademische Zwecke entwickelt.

---

## 👥 Kontakt

Bei Fragen oder Problemen, bitte ein Issue auf GitHub erstellen.

---

**Viel Erfolg beim Training! 🚀**

**WICHTIG**: Für beste Ergebnisse mit neuen Parametern **fresh training** starten (alte Checkpoints löschen)!
