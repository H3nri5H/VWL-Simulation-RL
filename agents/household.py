"""Haushalt mit regelbasiertem Verhalten"""
import numpy as np

class Household:
    """
    Haushalt konsumiert und spart basierend auf Einkommen.
    Kein RL - regelbasiertes Verhalten.
    """
    
    def __init__(self, household_id):
        self.id = household_id
        
        # Eigenschaften
        self.vermoegen = np.random.uniform(2000, 8000)
        self.sparquote = np.random.uniform(0.1, 0.3)  # 10-30% sparen
        self.konsumneigung = np.random.uniform(0.7, 0.95)  # 70-95% des Einkommens
        
        # Zustand
        self.einkommen = 0.0
        self.konsum = 0.0
    
    def receive_income(self, avg_lohn):
        """Einkommen basierend auf durchschnittlichem Lohn"""
        self.einkommen = avg_lohn
    
    def konsumieren(self):
        """Konsumentscheidung treffen"""
        # Konsum = Anteil des Einkommens + kleiner Teil des Vermögens
        self.konsum = self.einkommen * self.konsumneigung + \
                      self.vermoegen * 0.02  # 2% Vermögen
        
        # Sparen
        ersparnis = self.einkommen * self.sparquote
        self.vermoegen += ersparnis
        
        # Vermögen reduzieren wenn Konsum > Einkommen
        if self.konsum > self.einkommen:
            self.vermoegen -= (self.konsum - self.einkommen)
        
        self.vermoegen = max(0, self.vermoegen)  # Kein negatives Vermögen
        
        return self.konsum
