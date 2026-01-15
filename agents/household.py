import numpy as np

class Household:
    """Haushalt mit Konsum-, Spar- und Arbeitsverhalten"""
    
    def __init__(self, household_id):
        self.id = household_id
        # Zufällige Eigenschaften
        self.vermoegen = np.random.uniform(1000, 5000)
        self.einkommen = 0.0
        self.sparquote = np.random.uniform(0.05, 0.25)  # 5-25% sparen
        self.konsumneigung = np.random.uniform(0.6, 0.9)  # 60-90% konsumieren
        self.arbeitsproduktivitaet = np vp.random.uniform(0.8, 1.2)
        self.arbeitet = True
        
    def arbeiten(self, lohn):
        """Haushalt arbeitet und erhält Einkommen"""
        if self.arbeitet:
            self.einkommen = lohn * self.arbeitsproduktivitaet
            return self.arbeitsproduktivitaet
        return 0.0
    
    def konsumieren(self, preise):
        """Haushalt konsumiert basierend auf Einkommen und Vermögen"""
        verfuegbar = self.einkommen + self.vermoegen * 0.05  # 5% Vermögen nutzbar
        konsum = verfuegbar * self.konsumneigung
        
        # Sparen
        ersparnis = self.einkommen * self.sparquote
        self.vermoegen += ersparnis
        
        # Vermögen reduzieren durch Konsum
        self.vermoegen -= max(0, konsum - self.einkommen)
        self.vermoegen = max(0, self.vermoegen)
        
        return konsum
    
    def get_state(self):
        """Zustand des Haushalts"""
        return {
            'vermoegen': self.vermoegen,
            'einkommen': self.einkommen,
            'sparquote': self.sparquote
        }
