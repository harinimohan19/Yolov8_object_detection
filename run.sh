#!/bin/bash

echo "Starting FastAPI on port 8000..."
uvicorn main:app --host 0.0.0.0 --port 8000 &

echo "Starting Streamlit on port 8501..."
streamlit run app.py --server.address=0.0.0.0 --server.port=8501

wait
