@echo off
ECHO ===============================================
ECHO  Edge AI Predictive Maintenance -- Setup ^& Run
ECHO ===============================================

ECHO.
ECHO [1/5] Installing dependencies ...
pip install -r requirements.txt --quiet

ECHO.
ECHO [2/5] Generating synthetic datasets ...
python data\generate_data.py

ECHO.
ECHO [3/5] Training RUL prediction model (Aerospace) ...
python src\train_rul_model.py

ECHO.
ECHO [4/5] Training anomaly detection model (IHM) ...
python src\train_anomaly_model.py

ECHO.
ECHO [5/5] Running edge optimization ^& benchmarking ...
python src\edge_optimize.py

ECHO.
ECHO ===============================================
ECHO  Setup complete! Launching dashboard ...
ECHO ===============================================
streamlit run dashboard\app.py
