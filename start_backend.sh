#!/bin/bash

echo "========================================"
echo "Starting VWL-Simulation Backend"
echo "========================================"
echo ""

cd backend

echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

echo "Starting FastAPI server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

uvicorn app:app --reload --host 0.0.0.0 --port 8000
