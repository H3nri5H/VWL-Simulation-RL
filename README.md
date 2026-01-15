# Multi-Agent Volkswirtschafts-Simulation mit Reinforcement Learning

**Projektdauer:** 6 Wochen | **DHSH Fortgeschrittene KI-Anwendungen**

## 🎯 Projektziel

Simulation einer Volkswirtschaft wo **10 Unternehmen** durch Reinforcement Learning lernen:
- Optimale Preissetzung
- Produktionsmenge anpassen
- Löhne festlegen
- Auf Marktveränderungen reagieren

**Forschungsfragen:**
- Bilden sich Kartelle?
- Entstehen Preiskrämpfe?
- Wie reagieren Agents auf Wirtschaftsschocks?
- Können Agents nachhaltige Strategien lernen?

---

## 📅 Zeitplan

### **Woche 1-2: Foundation** ← AKTUELL
- [x] Projekt-Setup
- [ ] RLlib Multi-Agent Environment
- [ ] Erste Firma erfolgreich trainiert
- [ ] Bug-Fixes & Stabilisierung

### **Woche 3: Multi-Agent Training**
- [ ] 3-5 Firmen simultan trainieren
- [ ] Emergente Verhaltensweisen beobachten
- [ ] Reward-Tuning

### **Woche 4: Szenarien**
- [ ] Wirtschaftsschocks implementieren
- [ ] Robustheitstests
- [ ] Visualisierungen

### **Woche 5: Erweiterungen**
- [ ] Regierung als Agent (optional)
- [ ] Komplexere Interaktionen
- [ ] Feature-Erweiterungen

### **Woche 6: Analyse & Dokumentation**
- [ ] Experimente durchführen
- [ ] Ergebnisse auswerten
- [ ] Abschlussbericht

---

## 🚀 Quick Start

```bash
# Installation
git clone https://github.com/H3nri5H/VWL-Simulation-RL.git
cd VWL-Simulation-RL
pip install -r requirements.txt

# Erstes Training (1 Firma)
python train_single.py

# Multi-Agent Training
python train_marl.py

# Testen
python test.py
```

---

## 📋 Projektstruktur

```
VWL-Simulation-RL/
├── agents/               # Agent-Klassen
│   ├── firm.py          # Unternehmen (RL-Agent)
│   └── household.py     # Haushalt (regelbasiert)
├── envs/                # Environments
│   ├── economy_env.py   # PettingZoo MARL Environment
│   └── scenarios.py     # Wirtschaftsschocks
├── train_single.py      # 1 Firma trainieren
├── train_marl.py        # Multi-Agent Training
├── test.py              # Testen & Visualisierung
├── config.py            # Hyperparameter
└── utils/               # Helper-Funktionen
    ├── visualization.py
    └── metrics.py
```

---

## 🧠 Technologie-Stack

- **RLlib (Ray):** Multi-Agent RL Framework
- **PettingZoo:** Multi-Agent Environment Standard
- **TensorFlow:** Neural Networks
- **Gymnasium:** RL Environment API

---

## 📊 Nächste Schritte

1. **HEUTE:** RLlib Setup & Environment-Konvertierung
2. **Diese Woche:** Erste erfolgreiche Trainings
3. **Nächste Woche:** Multi-Agent Experimente starten

---

## 📄 Dokumentation

Siehe [docs/](docs/) für:
- Technische Dokumentation
- Experiment-Logs
- Ergebnisse & Analysen

---

**Status:** 🟡 In Entwicklung | **Letzte Aktualisierung:** 15.01.2026
