from typing import Dict
import pandas as pd
import joblib
import logging
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from challenge.config import (
    DATA_PATH,
    METRICS_OUTPUT_PATH,
    MODEL_OUTPUT_PATH,
    TARGET_COLUMN,
    VALIDATION_RATIO,
    RANDOM_STATE
)
from challenge.model import DelayModel


logging.basicConfig(level=logging.INFO)

def generate_classification_report(y_true, y_pred) -> Dict:
    """Generates a classification report as dict (for saving)."""
    report = classification_report(y_true, y_pred, output_dict=True)
    return report


def save_report(report: Dict, path: str) -> None:
    pd.DataFrame(report).transpose().to_json(path, indent=2)


def run_pipeline():
    # Load full dataset
    df = pd.read_csv(DATA_PATH)

    model = DelayModel()

    x, y = model.preprocess(df, target_column=TARGET_COLUMN)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=VALIDATION_RATIO, random_state=RANDOM_STATE)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_val)
    report = generate_classification_report(y_val, y_pred)

    logging.info("Saving classification report...")
    METRICS_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, METRICS_OUTPUT_PATH)

    # Retrain on full dataset for final model
    logging.info("Retraining model on full dataset...")
    model.fit(x, y)

    logging.info("Saving model...")
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUTPUT_PATH)

    logging.info("Pipeline completed.")


if __name__ == "__main__":
    run_pipeline()
