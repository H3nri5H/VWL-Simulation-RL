import os
import time
import io
import pandas as pd
import logging
import dotenv
from sqlalchemy import create_engine

# Logging-Konfiguration für besseres Feedback bei großen Dateien
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Konfiguration
dotenv.load_dotenv()  # .env-Datei laden, falls vorhanden
DATA_PATH = os.getenv("DATA_PATH", "./simulation_results/")
DB_URL = os.getenv("DATABASE_URL")

engine = create_engine(DB_URL)


def upload(df, table_name, engine):
    """
    Fast Upload mittels COPY-Befehl. Sehr viel schneller als to_sql, besonders bei großen Datenmengen.
    """
    # Spaltenreihenfolge sicherstellen (muss exakt der DB-Tabelle entsprechen)
    if table_name == "firms":
        cols = [
            "seed",
            "step",
            "firm_id",
            "price",
            "wage",
            "capital",
            "employees",
            "max_employees",
            "quality",
            "marketing",
            "profit",
            "revenue",
            "costs",
            "inventory",
            "production",
            "sales",
            "bankrupt",
        ]
    else:  # households
        cols = [
            "seed",
            "step",
            "household_id",
            "money",
            "employer",
            "wage",
            "skill_level",
            "wealth_type",
        ]

    # DataFrame auf diese Spalten einschränken/sortieren
    df = df[cols]

    raw_conn = engine.raw_connection()
    try:
        cursor = raw_conn.cursor()
        # Daten in einen In-Memory Buffer schreiben
        output = io.StringIO()
        df.to_csv(output, sep="\t", header=False, index=False)
        output.seek(0)

        # Der COPY-Befehl streamt die Daten direkt in die DB
        cursor.copy_from(output, table_name, sep="\t", null="")
        raw_conn.commit()
        return True
    except Exception as e:
        raw_conn.rollback()
        logging.error(f"Fehler beim COPY-Upload: {e}")
        return False
    finally:
        cursor.close()
        raw_conn.close()


def process():
    files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

    if not files:
        return

    for file_name in sorted(files):
        file_path = os.path.join(DATA_PATH, file_name)
        table_name = "firms" if "firms" in file_name else "households"
        start_time = time.time()

        try:
            logging.info(f"📂 Verarbeite {file_name}...")
            df = pd.read_csv(file_path)

            # Datentypen korrigieren
            if table_name == "firms" and "bankrupt" in df.columns:
                df["bankrupt"] = df["bankrupt"].astype(bool)

            # Turbo-Upload statt to_sql
            success = upload(df, table_name, engine)

            if success:
                os.remove(file_path)
                duration = time.time() - start_time
                logging.info(
                    f"✅ Erledigt: {len(df)} Zeilen in {duration:.2f}s hochgeladen und gelöscht."
                )
            else:
                logging.warning(
                    f"⚠️ Upload fehlgeschlagen für {file_name}. Datei bleibt liegen."
                )

        except Exception as e:
            logging.error(f"❌ Kritischer Fehler bei {file_name}: {e}")


if __name__ == "__main__":
    logging.info("🚀 Uploader-Container gestartet. Warte auf Daten...")
    while True:
        process()
        time.sleep(10)
