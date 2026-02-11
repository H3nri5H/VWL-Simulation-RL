import os
import time
import pandas as pd
from sqlalchemy import create_engine

# Konfiguration (wird über Docker-Env-Variablen übergeben)
DATA_PATH = "/app/data"
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@db-proxy:5432/simulation_db")

engine = create_engine(DB_URL)


def process_and_cleanup():
    # Alle CSV-Dateien im geteilten Ordner finden
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

    if not files:
        return

    for file_name in sorted(files):
        file_path = os.path.join(DATA_PATH, file_name)

        # Bestimmen, in welche Tabelle die Daten müssen
        table_name = "firms" if "firms" in file_name else "households"

        try:
            # 1. Daten lesen
            df = pd.read_csv(file_path)

            # 2. Datentypen korrigieren (für Postgres wichtig)
            if table_name == "firms" and "bankrupt" in df.columns:
                df["bankrupt"] = df["bankrupt"].astype(bool)

            # 3. In Datenbank schreiben
            # method='multi' beschleunigt den Prozess massiv
            df.to_sql(
                table_name, engine, if_exists="append", index=False, method="multi"
            )

            # 4. Löschen nach Erfolg
            os.remove(file_path)
            print(f"✅ Datei erfolgreich verarbeitet und gelöscht: {file_name}")

        except Exception as e:
            print(f"❌ Fehler bei {file_name}: {e}")
            print("Datei bleibt für den nächsten Versuch im Ordner.")


if __name__ == "__main__":
    print("🚀 Uploader-Container gestartet. Warte auf Daten...")
    while True:
        process_and_cleanup()
        time.sleep(10)  # Alle 10 Sekunden nach neuen Dateien suchen
