"""Unternehmen als RL-Agent mit vereinfachtem Action Space"""
import numpy as np

class Firm:
    """
    Unternehmen lernt durch RL:
    - Preis setzen
    - Produktionsmenge anpassen (implizit durch Mitarbeiter)
    - Löhne festlegen
    """
    
    def __init__(self, firm_id):
        self.id = firm_id
        
        # === EIGENSCHAFTEN (konstant) ===
        self.produktivitaet = np.random.uniform(0.8, 1.2)  # Wie effizient produziert wird
        self.fixkosten = np.random.uniform(500, 1000)  # Monatliche Fixkosten
        
        # === ZUSTAND (veränderlich) ===
        self.kapital = np.random.uniform(10000, 20000)
        self.preis = 100.0  # Startpreis
        self.lohn = 50.0    # Startlohn
        self.mitarbeiter = 10  # Startmitarbeiter
        
        # === PERFORMANCE ===
        self.produktion = 0.0
        self.verkauft = 0.0
        self.umsatz = 0.0
        self.gewinn = 0.0
        self.marktanteil = 0.0
        
        # History
        self.gewinn_history = []
        self.preis_history = []
    
    def apply_action(self, action):
        """
        Action Space: [preis_faktor, lohn_faktor, mitarbeiter_change]
        - preis_faktor: 0.8 - 1.2 (Preis um ±20% ändern)
        - lohn_faktor: 0.9 - 1.1 (Lohn um ±10% ändern)
        - mitarbeiter_change: -5 bis +5
        """
        preis_faktor, lohn_faktor, mitarbeiter_change = action
        
        # Preis anpassen
        self.preis *= preis_faktor
        self.preis = np.clip(self.preis, 50, 200)  # Realistischer Bereich
        
        # Lohn anpassen
        self.lohn *= lohn_faktor
        self.lohn = np.clip(self.lohn, 30, 100)  # Mindest-/Maximallohn
        
        # Mitarbeiter anpassen
        self.mitarbeiter += int(mitarbeiter_change)
        self.mitarbeiter = max(0, min(self.mitarbeiter, 50))  # 0-50 Mitarbeiter
    
    def produzieren(self):
        """Cobb-Douglas Produktionsfunktion: Y = A * K^0.3 * L^0.7"""
        if self.mitarbeiter > 0 and self.kapital > 0:
            self.produktion = self.produktivitaet * \
                (self.kapital ** 0.3) * \
                (self.mitarbeiter ** 0.7)
        else:
            self.produktion = 0.0
        return self.produktion
    
    def berechne_kosten(self):
        """Gesamtkosten berechnen"""
        lohnkosten = self.mitarbeiter * self.lohn
        return self.fixkosten + lohnkosten
    
    def update_finanzen(self, verkaufte_menge):
        """Finanzen nach Verkauf aktualisieren"""
        self.verkauft = verkaufte_menge
        self.umsatz = self.verkauft * self.preis
        kosten = self.berechne_kosten()
        self.gewinn = self.umsatz - kosten
        
        # Kapital anpassen (70% des Gewinns reinvestieren)
        self.kapital += self.gewinn * 0.7
        self.kapital = max(1000, self.kapital)  # Mindestkapital
        
        # History speichern
        self.gewinn_history.append(self.gewinn)
        self.preis_history.append(self.preis)
    
    def get_observation(self, markt_info):
        """
        Observation für RL-Agent (8 Features):
        - Eigene Metriken: Kapital, Preis, Lohn, Mitarbeiter, Gewinn
        - Markt: Durchschnittspreis, Nachfrage, Angebot
        """
        return np.array([
            self.kapital / 20000,           # Normalisiert
            self.preis / 100,               # Normalisiert
            self.lohn / 50,                 # Normalisiert
            self.mitarbeiter / 50,          # Normalisiert
            self.gewinn / 1000,             # Normalisiert
            markt_info['avg_preis'] / 100,  # Durchschnittspreis
            markt_info['nachfrage'] / 1000, # Gesamtnachfrage
            markt_info['angebot'] / 1000    # Gesamtangebot
        ], dtype=np.float32)
    
    def calculate_reward(self):
        """
        Reward-Funktion:
        - Gewinn (Hauptziel)
        - Marktanteil (langfristige Position)
        - Stabilität (Kapitaldecke)
        """
        # Gewinn-Reward (normalisiert)
        gewinn_reward = np.clip(self.gewinn / 100, -10, 10)
        
        # Marktanteil-Reward
        marktanteil_reward = self.marktanteil * 5
        
        # Kapital-Penalty bei niedrigem Kapital
        if self.kapital < 5000:
            kapital_penalty = -5
        else:
            kapital_penalty = 0
        
        return gewinn_reward + marktanteil_reward + kapital_penalty
    
    def get_state(self):
        """Dict für Logging"""
        return {
            'id': self.id,
            'kapital': round(self.kapital, 0),
            'preis': round(self.preis, 1),
            'lohn': round(self.lohn, 1),
            'mitarbeiter': self.mitarbeiter,
            'produktion': round(self.produktion, 1),
            'verkauft': round(self.verkauft, 1),
            'gewinn': round(self.gewinn, 0),
            'marktanteil': round(self.marktanteil * 100, 1)
        }
