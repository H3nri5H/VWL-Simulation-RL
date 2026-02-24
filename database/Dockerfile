FROM python:3.11-slim

# System-Abhängigkeiten für PostgreSQL installieren
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Abhängigkeiten installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Skript kopieren
COPY . .
# Die .env wird über Docker-Compose geladen, nicht fest einkopiert (sicherer)

CMD ["python", "data_upload.py"]