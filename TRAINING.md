# Training Guide - VWL-Simulation-RL

Anleitung zum Trainieren der Multi-Agent KI für die Volkswirtschaftssimulation.

## 🚀 Quick Start

### Windows (WSL2 empfohlen)

**Option A: Mit WSL2 (empfohlen für stabiles Training)**

```bash
# 1. WSL2 aktivieren (falls noch nicht)
wsl --install

# 2. In WSL2:
cd /mnt/c/Users/DEIN_USER/VWL-Simulation-RL
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Training starten
python train.py
```

**Option B: Native Windows (kann DLL-Probleme haben)**

```bash
# Nur wenn WSL2 nicht möglich
venv\Scripts\activate
python train.py

# Falls DLL-Fehler: Nutze WSL2 oder Cloud!
```

### Linux/Mac

```bash
# 1. Setup
source venv/bin/activate
pip install -r requirements.txt

# 2. Training starten
python train.py
```

---

## 🎯 Training Modes

### 1. Quick Test (10 Iterationen, ~5 Minuten)

```bash
python train.py --iterations 10
```

**Nutzen:**
- Testet ob alles funktioniert
- Erste Rewards sehen
- Kein sinnvolles Modell!

### 2. Short Training (100 Iterationen, ~30-60 Minuten)

```bash
python train.py --iterations 100 --checkpoint-freq 20
```

**Nutzen:**
- Erste sinnvolle Policy
- Firmen lernen Basics (Preise anpassen, Löhne setzen)
- Gut für erste Tests im Dashboard

### 3. Full Training (500-1000 Iterationen, ~3-8 Stunden)

```bash
python train.py --iterations 500 --checkpoint-freq 50
```

**Nutzen:**
- Stabile Policy
- Firmen finden Gleichgewicht
- Produktionsreif für Präsentation

### 4. Production Training (2000+ Iterationen, über Nacht)

```bash
nohup python train.py --iterations 2000 --checkpoint-freq 100 > training.log 2>&1 &

# Log verfolgen:
tail -f training.log
```

**Nutzen:**
- Optimale Policy
- Komplexe Strategien
- Benchmark-Qualität

---

## ⚙️ Parameter-Übersicht

### Environment Parameters

```bash
python train.py \
  --n-firms 2 \          # Anzahl Firmen (default: 2)
  --n-households 10      # Anzahl Haushalte (default: 10)
```

**Achtung:** Mehr Firmen/Haushalte = längeres Training!

### Training Parameters

```bash
python train.py \
  --iterations 100 \           # Trainings-Iterationen
  --learning-rate 0.0003 \     # Lernrate (default: 3e-4)
  --num-workers 2 \            # Parallele Rollout-Workers
  --checkpoint-freq 10         # Checkpoint alle N Iterationen
```

**Learning Rate Tipps:**
- `3e-4` (default): Stabil, langsam
- `1e-3`: Schneller, aber instabiler
- `1e-4`: Sehr stabil, sehr langsam

**Num Workers:**
- `0`: Single-threaded (debugging)
- `2-4`: Gut für Laptops
- `8+`: Server/Cloud

---

## 📊 Training Output verstehen

### Beispiel-Output:

```
============================================================
VWL-Simulation Multi-Agent RL Training
============================================================
Environment Config:
  - Firms: 2
  - Households: 10

Training Config:
  - Algorithm: PPO (Proximal Policy Optimization)
  - Framework: TensorFlow 2.x
  - Learning Rate: 0.0003
  - Workers: 2
  - Total Iterations: 100
  - Checkpoint Every: 10 iterations
============================================================

[Iteration 1/100]
  Reward: -45.23 (min: -120.50, max: 30.20)
  Episode Length: 100.0

[Iteration 10/100]
  Reward: 85.30 (min: 20.10, max: 150.40)
  Episode Length: 100.0
  ✅ Checkpoint saved: /home/user/ray_results/PPO_2026-01-31.../checkpoint_000010

...

[Iteration 100/100]
  Reward: 250.80 (min: 180.30, max: 320.50)
  Episode Length: 100.0
  ✅ Checkpoint saved: /home/user/ray_results/PPO_2026-01-31.../checkpoint_000100
  🏆 New best reward: 250.80

============================================================
Training Complete!
============================================================
Final checkpoint saved: /home/user/ray_results/PPO_2026-01-31.../checkpoint_000100
Best reward achieved: 250.80

✅ Checkpoint copied to: models/checkpoint_000100
   You can now use this in the dashboard!
```

### Reward Interpretation:

**Negativ (-50 bis 0):**
- Firmen machen Verluste
- Schlechte Preis/Lohn-Kombination
- Learning in Progress...

**Niedrig (0 bis 100):**
- Firmen überleben
- Erste Profite
- Noch suboptimal

**Mittel (100 bis 200):**
- Stabile Profite
- Gute Strategien
- Nutzbar!

**Hoch (200+):**
- Optimale Strategien
- Markt-Gleichgewicht
- Produktionsreif ✅

---

## 💾 Checkpoints nutzen

### Wo werden Checkpoints gespeichert?

```bash
# Ray speichert in:
~/ray_results/PPO_EconomyEnv_YYYY-MM-DD_HH-MM-SS/

# Automatisch kopiert nach:
models/checkpoint_XXXXXX/
```

### Checkpoint im Dashboard nutzen:

1. **Backend starten:**
   ```bash
   start_backend.bat  # oder start_backend.sh
   ```

2. **Frontend starten:**
   ```bash
   start_frontend.bat  # oder start_frontend.sh
   ```

3. **Im Dashboard:**
   - Modell auswählen: `checkpoint_000100` (statt "random")
   - Simulation starten
   - Trainierte Policy analysieren!

### Checkpoint manuell kopieren:

```bash
# Falls automatisches Kopieren fehlschlägt:
cp -r ~/ray_results/PPO_*/checkpoint_000100 models/

# Windows:
xcopy /E /I %USERPROFILE%\ray_results\PPO_*\checkpoint_000100 models\checkpoint_000100
```

---

## 🐛 Troubleshooting

### Problem: "DLL load failed" (Windows)

**Lösung:** Nutze WSL2 oder Cloud!

```bash
# In WSL2:
wsl
cd /mnt/c/Users/DEIN_USER/VWL-Simulation-RL
source venv/bin/activate
python train.py
```

### Problem: "Ray failed to start"

**Lösung 1:** Reduziere Workers

```bash
python train.py --num-workers 0
```

**Lösung 2:** Ray neu installieren

```bash
pip uninstall ray
pip install ray[rllib]==2.10.0
```

### Problem: "CUDA not available" Warnung

**Ignorieren!** Wir nutzen CPU (besser für Windows).

### Problem: Training sehr langsam

**Lösung 1:** Mehr Workers (falls CPU-Kerne vorhanden)

```bash
python train.py --num-workers 4
```

**Lösung 2:** Kleinere Batch-Size (Code-Änderung nötig)

**Lösung 3:** Cloud-Training (siehe unten)

### Problem: Reward steigt nicht

**Mögliche Ursachen:**
- Zu wenig Iterationen (< 50)
- Learning Rate zu hoch/niedrig
- Environment-Bug

**Debugging:**

```bash
# 1. Environment manuell testen:
python test_env.py

# 2. Mit niedrigerer Learning Rate:
python train.py --learning-rate 1e-4

# 3. Mehr Iterationen:
python train.py --iterations 200
```

---

## ☁️ Cloud Training (Google Colab/Cloud)

Falls lokales Training zu langsam oder Windows-Probleme:

### Google Colab (kostenlos):

```python
# In Colab-Notebook:
!git clone https://github.com/H3nri5H/VWL-Simulation-RL.git
%cd VWL-Simulation-RL
!pip install -r requirements.txt

# Training starten:
!python train.py --iterations 500

# Checkpoints downloaden:
from google.colab import files
import shutil
shutil.make_archive('checkpoints', 'zip', 'models')
files.download('checkpoints.zip')
```

### Google Cloud VM:

Siehe separate Dokumentation (coming soon).

---

## 📊 Monitoring während Training

### Terminal-Output verfolgen:

```bash
# Bei nohup:
tail -f training.log

# Live-Statistiken:
watch -n 5 'tail -20 training.log'
```

### TensorBoard (optional):

```bash
# In separatem Terminal:
tensorboard --logdir ~/ray_results

# Browser: http://localhost:6006
```

---

## ✅ Checkliste: Bereit zum Trainieren?

- [ ] Python 3.10 oder 3.11 installiert
- [ ] venv aktiviert
- [ ] `pip install -r requirements.txt` erfolgreich
- [ ] `python test_env.py` funktioniert
- [ ] WSL2 (Windows) oder Linux/Mac
- [ ] Genug Zeit (1-8 Stunden je nach Iterationen)

**Los geht's:**

```bash
python train.py --iterations 100
```

🎉 Viel Erfolg beim Training! Bei Fragen: Issue auf GitHub öffnen.
