# VWL-Simulation mit Multi-Agent Reinforcement Learning

## 🎯 Projektübersicht

Volkswirtschafts-Simulation mit Reinforcement Learning für das Modul **Fortgeschrittene KI-Anwendungen** (DHSH).

### Agents:
- **10 Unternehmen** (RL-Agents): Lernen Preissetzung, Produktionsmenge, Löhne
- **1 Regierung** (RL-Agent): Lernt Steuerpolitik, Staatsausgaben, Leitzins
- **50 Haushalte** (Regelbasiert): Konsum, Sparen, Arbeiten

## 🚀 Quick Start

```bash
# Installation
git clone https://github.com/H3nri5H/VWL-Simulation-RL.git
cd VWL-Simulation-RL
pip install -r requirements.txt

# Training (Regierung mit Random Firmen)
python train_marl.py

# Testen
python test_marl.py
```

## 📋 Struktur

```
VWL-Simulation-RL/
├── agents/
│   ├── household.py      # Haushalte (regelbasiert)
│   └── firm.py           # Unternehmen (RL-Agent)
├── marl_economy_env.py   # Haupt-Environment
├── train_marl.py         # Training
├── test_marl.py          # Testing
└── requirements.txt
```

## 🧠 Konzept

### Unternehmen (RL-Agents)
**Observation Space (10 Features):**
- Eigenes Kapital, Preis, Lohn, Mitarbeiter
- Eigener Gewinn, Marktanteil
- Markt: Durchschnittspreis, Nachfrage, Angebot, Arbeitslosenquote

**Action Space:**
- Preisänderung (-10% bis +10%)
- Mitarbeiteränderung (-5 bis +5)
- Lohnänderung (-5% bis +5%)

**Reward:**
- Gewinn (Hauptziel)
- Marktanteil (langfristig)
- Kapitalstabilität
- Effizienz (verkauft/produziert)
- Beschäftigung (soziale Verantwortung)

### Regierung (RL-Agent)
**Observation:** BIP, Arbeitslosigkeit, Inflation, Schulden

**Action:** Steuersatz, Staatsausgaben, Leitzins

**Reward:** BIP-Wachstum, niedrige Inflation, niedrige Arbeitslosigkeit

## ⚠️ Limitation

Die aktuelle Implementierung nutzt **Stable-Baselines3**, das kein natives Multi-Agent Learning unterstützt.

### Für echtes MARL:
1. **RLlib** (Ray): Industry-Standard für MARL
2. **PettingZoo**: Multi-Agent Gym-Wrapper

Aktuell: Regierung wird trainiert, Firmen machen Random Actions.

## 📈 Nächste Schritte

1. Integration mit RLlib für echtes Multi-Agent Learning
2. Szenarien: Steuererhöhung, Rezession, Naturkatastrophen
3. Analyse: Kartellbildung, Preiskampf, Marktkonzentration
4. Erweitern: Banken, Zentralbank, Export/Import

## 📚 Quellen

- **Cobb-Douglas Produktion**: Standard-Ökonometrie
- **Phillips-Kurve**: Inflation-Arbeitslosigkeit Trade-off
- **Multi-Agent RL**: Emergente Verhaltensweisen in Wirtschaftssystemen
