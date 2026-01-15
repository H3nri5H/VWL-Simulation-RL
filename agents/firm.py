import numpy as np

class Firm:
    """Unternehmen mit Produktion, Preissetzung und Beschäftigung"""
    
    def __init__(self, firm_id):
        self.id = firm_id
        # Zufällige Eigenschaften
        self.kapital = np.random.uniform(5000, 15000)
        self.produktivitaet = np.random.uniform(0.7, 1.3)
        self.lohn = np.random.uniform(40, 60)
        self.preis = np.random.uniform(80, 120)
        self.fixkosten = np.random.uniform(500, 1500)
        self.variable_kosten_pro_einheit = np.random.uniform(30, 50)
        
        self.mitarbeiter = 0
        self.produktion = 0.0
        self.umsatz = 0.0
        self.gewinn = 0.0
        
    def einstellen(self, arbeitskraefte_angebot, max_mitarbeiter=50):
        """Unternehmen stellt Arbeitskräfte ein"""
        # Einfache Regel: Stelle ein wenn profitabel
        gewuenschte_mitarbeiter = min(
            int(self.kapital / (self.lohn * 12)),  # Kann 12 Monate Lohn zahlen
            max_mitarbeiter
        )
        self.mitarbeiter = min(gewuenschte_mitarbeiter, arbeitskraefte_angebot)
        return self.mitarbeiter
    
    def produzieren(self):
        """Unternehmen produziert Güter"""
        # Cobb-Douglas: Y = A * K^0.4 * L^0.6
        if self.mitarbeiter > 0:
            self.produktion = self.produktivitaet * \
                (self.kapital ** 0.4) * \
                (self.mitarbeiter ** 0.6)
        else:
            self.produktion = 0.0
        return self.produktion
    
    def verkaufen(self, nachfrage_gesamt, angebot_gesamt):
        """Unternehmen verkauft Produktion am Markt"""
        if angebot_gesamt > 0:
            # Marktanteil basierend auf Produktion
            marktanteil = self.produktion / angebot_gesamt
            verkauft = min(self.produktion, nachfrage_gesamt * marktanteil)
        else:
            verkauft = 0.0
        
        self.umsatz = verkauft * self.preis
        
        # Kosten berechnen
        lohnkosten = self.mitarbeiter * self.lohn
        variable_kosten = verkauft * self.variable_kosten_pro_einheit
        gesamtkosten = self.fixkosten + lohnkosten + variable_kosten
        
        self.gewinn = self.umsatz - gesamtkosten
        
        # Kapital anpassen
        self.kapital += self.gewinn * 0.3  # 30% reinvestieren
        self.kapital = max(1000, self.kapital)  # Minimum
        
        return verkauft
    
    def preis_anpassen(self, nachfrage, angebot):
        """Einfache Preisanpassung basierend auf Angebot/Nachfrage"""
        if angebot > 0:
            verhaeltnis = nachfrage / angebot
            if verhaeltnis > 1.1:  # Nachfrage > Angebot
                self.preis *= 1.05  # +5%
            elif verhaeltnis < 0.9:  # Angebot > Nachfrage
                self.preis *= 0.95  # -5%
        
        self.preis = np.clip(self.preis, 50, 200)  # Grenzen
    
    def get_state(self):
        """Zustand des Unternehmens"""
        return {
            'kapital': self.kapital,
            'mitarbeiter': self.mitarbeiter,
            'produktion': self.produktion,
            'gewinn': self.gewinn,
            'preis': self.preis
        }
