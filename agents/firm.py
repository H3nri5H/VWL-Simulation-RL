import numpy as np

class Firm:
    """
    Unternehmen als RL-Agent.
    Lernt: Preissetzung, Produktionsmenge, Löhne
    
    Jedes Unternehmen hat individuelle Eigenschaften und lernt
    durch Reinforcement Learning optimale Strategien.
    """
    
    def __init__(self, firm_id):
        self.id = firm_id
        
        # === EIGENSCHAFTEN (fix während Simulation) ===
        # Produktivität: Wie effizient das Unternehmen produziert
        self.produktivitaet = np.random.uniform(0.7, 1.3)
        
        # Fixkosten: Müssen jeden Monat bezahlt werden (Miete, etc.)
        self.fixkosten = np.random.uniform(500, 1500)
        
        # Variable Kosten pro Einheit: Material, Energie, etc.
        self.variable_kosten_pro_einheit = np.random.uniform(30, 50)
        
        # === ZUSTANDSVARIABLEN (verändern sich) ===
        # Startkapital des Unternehmens
        self.kapital = np.random.uniform(5000, 15000)
        
        # Aktueller Preis für Produkte
        self.preis = np.random.uniform(80, 120)
        
        # Aktueller Lohn für Mitarbeiter
        self.lohn = np.random.uniform(40, 60)
        
        # Anzahl beschäftigter Mitarbeiter
        self.mitarbeiter = 0
        
        # === PERFORMANCE-METRIKEN ===
        self.produktion = 0.0  # Produzierte Einheiten
        self.verkauft = 0.0     # Verkaufte Einheiten
        self.umsatz = 0.0       # Einnahmen
        self.gewinn = 0.0       # Profit
        self.marktanteil = 0.0  # Anteil am Gesamtmarkt
        
        # History für Lernkurve
        self.gewinn_history = []
        self.marktanteil_history = []
    
    def get_observation(self, markt_info):
        """
        Observation Space für RL-Agent:
        - Eigener Zustand (Kapital, Preis, Mitarbeiter, etc.)
        - Marktinformationen (Durchschnittspreise, Nachfrage, etc.)
        
        Returns: numpy array mit 10 Features
        """
        return np.array([
            self.kapital / 10000,              # 0: Normalisiertes Kapital
            self.preis / 100,                   # 1: Normalisierter Preis
            self.lohn / 50,                     # 2: Normalisierter Lohn
            self.mitarbeiter / 50,              # 3: Normalisierte Mitarbeiteranzahl
            self.gewinn / 1000,                 # 4: Normalisierter Gewinn
            self.marktanteil,                   # 5: Marktanteil (0-1)
            markt_info['avg_preis'] / 100,      # 6: Durchschnittspreis am Markt
            markt_info['nachfrage'] / 10000,    # 7: Gesamtnachfrage
            markt_info['angebot'] / 10000,      # 8: Gesamtangebot
            markt_info['arbeitslosenquote']     # 9: Arbeitslosenquote
        ], dtype=np.float32)
    
    def apply_action(self, action, max_mitarbeiter=50):
        """
        RL-Agent wählt Aktion:
        action[0]: Preisänderung (-1 bis +1, wird zu -10% bis +10%)
        action[1]: Mitarbeiteränderung (-1 bis +1)
        action[2]: Lohnänderung (-1 bis +1, wird zu -5% bis +5%)
        """
        # === PREIS ANPASSEN ===
        # Action zwischen -1 und +1, wird zu -10% bis +10%
        preis_aenderung = action[0] * 0.10
        self.preis *= (1 + preis_aenderung)
        # Preis muss zwischen 50 und 200 bleiben (realistisch)
        self.preis = np.clip(self.preis, 50, 200)
        
        # === MITARBEITER ANPASSEN ===
        # Action zwischen -1 und +1, wird zu -5 bis +5 Mitarbeiter
        mitarbeiter_aenderung = int(action[1] * 5)
        self.mitarbeiter += mitarbeiter_aenderung
        # Kann nicht mehr einstellen als bezahlbar oder max_mitarbeiter
        max_bezahlbar = int(self.kapital / (self.lohn * 12))  # 12 Monate Lohn
        self.mitarbeiter = np.clip(self.mitarbeiter, 0, min(max_mitarbeiter, max_bezahlbar))
        
        # === LOHN ANPASSEN ===
        # Action zwischen -1 und +1, wird zu -5% bis +5%
        lohn_aenderung = action[2] * 0.05
        self.lohn *= (1 + lohn_aenderung)
        # Mindestlohn 30, Maximallohn 100
        self.lohn = np.clip(self.lohn, 30, 100)
    
    def produzieren(self):
        """
        Produktionsfunktion: Cobb-Douglas
        Y = A * K^0.4 * L^0.6
        
        A = Produktivität
        K = Kapital
        L = Arbeitskräfte (Mitarbeiter)
        """
        if self.mitarbeiter > 0:
            self.produktion = self.produktivitaet * \
                (self.kapital ** 0.4) * \
                (self.mitarbeiter ** 0.6)
        else:
            self.produktion = 0.0
        
        return self.produktion
    
    def verkaufen(self, nachfrage_gesamt, angebot_gesamt, alle_preise):
        """
        Verkauf am Markt mit Preiswettbewerb:
        - Günstigere Firmen verkaufen mehr
        - Marktanteil hängt von relativem Preis ab
        
        Returns: Anzahl verkaufter Einheiten
        """
        if angebot_gesamt == 0 or self.produktion == 0:
            self.verkauft = 0.0
            self.umsatz = 0.0
            self.gewinn = -self.fixkosten
            return 0.0
        
        # === MARKTANTEIL BASIEREND AUF PREIS ===
        # Günstigere Firmen bekommen höheren Marktanteil
        avg_preis = np.mean(alle_preise)
        
        # Preiswettbewerbsfaktor: Je günstiger, desto höher
        # Wenn Preis = avg_preis -> Faktor = 1.0
        # Wenn Preis < avg_preis -> Faktor > 1.0 (mehr Marktanteil)
        # Wenn Preis > avg_preis -> Faktor < 1.0 (weniger Marktanteil)
        preis_wettbewerb_faktor = (avg_preis / self.preis) ** 2
        
        # Kapazitätsanteil: Wie viel kann die Firma liefern?
        kapazitaets_anteil = self.produktion / angebot_gesamt
        
        # Kombinierter Marktanteil
        gewichteter_anteil = kapazitaets_anteil * preis_wettbewerb_faktor
        
        # Normalisierung kommt später durch alle Firmen
        # Hier nur: Was könnte maximal verkauft werden?
        max_verkauf = min(self.produktion, nachfrage_gesamt * gewichteter_anteil)
        self.verkauft = max_verkauf
        
        # === FINANZEN BERECHNEN ===
        # Umsatz = Verkaufte Einheiten * Preis
        self.umsatz = self.verkauft * self.preis
        
        # Kosten berechnen
        lohnkosten = self.mitarbeiter * self.lohn
        variable_kosten = self.verkauft * self.variable_kosten_pro_einheit
        gesamtkosten = self.fixkosten + lohnkosten + variable_kosten
        
        # Gewinn = Umsatz - Kosten
        self.gewinn = self.umsatz - gesamtkosten
        
        # === KAPITAL ANPASSEN ===
        # Gewinn wird teilweise reinvestiert (70%)
        self.kapital += self.gewinn * 0.7
        
        # Unternehmen kann nicht unter Mindestkapital fallen
        self.kapital = max(1000, self.kapital)
        
        # Bei negativem Kapital: Insolvenzrisiko
        if self.kapital < 2000:
            # Notverkauf von Assets / Kreditaufnahme
            self.kapital = 2000
            self.gewinn -= 500  # Strafkosten
        
        # History speichern
        self.gewinn_history.append(self.gewinn)
        self.marktanteil_history.append(self.marktanteil)
        
        return self.verkauft
    
    def calculate_reward(self, markt_info):
        """
        Reward-Funktion für RL-Agent:
        - Gewinn (Hauptziel)
        - Marktanteil (langfristige Position)
        - Kapitalstabilität
        
        Returns: float reward
        """
        # === KOMPONENTEN ===
        # 1. Gewinn-Reward (normalisiert)
        gewinn_reward = self.gewinn / 100
        
        # 2. Marktanteil-Reward (langfristige Position)
        marktanteil_reward = self.marktanteil * 50
        
        # 3. Kapitalstabilität (bestraft niedrige Kapitaldecke)
        if self.kapital < 3000:
            kapital_penalty = -10
        else:
            kapital_penalty = 0
        
        # 4. Effizienz-Bonus (verkauft alles was produziert wird)
        if self.produktion > 0:
            effizienz = self.verkauft / self.produktion
            effizienz_bonus = effizienz * 5
        else:
            effizienz_bonus = 0
        
        # 5. Beschäftigung-Bonus (soziale Verantwortung)
        beschaftigung_bonus = self.mitarbeiter * 0.2
        
        # === GESAMT-REWARD ===
        total_reward = (
            gewinn_reward +
            marktanteil_reward +
            kapital_penalty +
            effizienz_bonus +
            beschaftigung_bonus
        )
        
        return total_reward
    
    def get_state(self):
        """Zustand für Logging/Debugging"""
        return {
            'id': self.id,
            'kapital': self.kapital,
            'preis': self.preis,
            'lohn': self.lohn,
            'mitarbeiter': self.mitarbeiter,
            'produktion': self.produktion,
            'verkauft': self.verkauft,
            'gewinn': self.gewinn,
            'marktanteil': self.marktanteil,
            'umsatz': self.umsatz
        }
