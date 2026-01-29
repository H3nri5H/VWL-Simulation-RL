#!/bin/bash

echo "========================================"
echo "Starting VWL-Simulation Frontend"
echo "========================================"
echo ""

cd frontend

echo "Installing dependencies..."
pip install -r requirements.txt
echo ""

echo "Starting Streamlit on http://localhost:8501"
echo "Browser will open automatically..."
echo "Press Ctrl+C to stop"
echo ""

streamlit run dashboard.py
