from datetime import datetime
from pathlib import Path

# Data
DATA_PATH = Path("data/data.csv")
ORIGINAL_COLS = ["OPERA", "TIPOVUELO", "MES"]
TARGET_COLUMN = "delay"

# Artifacts
MODEL_OUTPUT_PATH = Path(f"models/delay_model_{datetime.now().strftime('%Y%m%dT%H')}.joblib")
METRICS_OUTPUT_PATH = Path(f"metrics/classification_report_{datetime.now().strftime('%Y%m%dT%H')}.json")

# Evaluation
VALIDATION_RATIO = 0.2
RANDOM_STATE = 42

# API MODEL
API_MODEL_PATH = Path("models/delay_model_20250514T19.joblib")
