import logging
from typing import Optional
import joblib
import pandas as pd
from fastapi import Request, HTTPException, FastAPI

from challenge.config import API_MODEL_PATH
from challenge.model import DelayModel
from challenge.schemas import PredictionRequest, PredictionResponse

app = FastAPI()

model: Optional[DelayModel] = None
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@app.on_event("startup")
async def startup_event_load_model():
    """
    Load the machine learning model when the application starts.
    """
    global model
    try:
        model = joblib.load(API_MODEL_PATH)
        logger.info(f"Model loaded successfully from {API_MODEL_PATH}")
    except FileNotFoundError as e:
        logger.error(f"Model file not found at {API_MODEL_PATH}. The application will start without a loaded model.")
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.") from e
    except Exception as e:
        logger.error(f"Error loading model from {API_MODEL_PATH}: {e}")
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.") from e


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: Request) -> PredictionResponse:
    """
    Predict delay based on flight info.
    """
    try:
        flights_json = await request.json()
        validated_request = PredictionRequest(**flights_json["flights"][0])

        if validated_request.OPERA not in model._trained_airlines:
            raise ValueError("Invalid airline.")

        input_df = pd.DataFrame([validated_request.dict()])
        features = model.preprocess(input_df)
        prediction = model.predict(features)

        return PredictionResponse(predict=prediction)
    except ValueError as e:
        logger.error("At least one field is invalid: %s", str(e))
        raise HTTPException(status_code=400, detail='Invalid input.') from e
    except Exception as e:
        logger.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed.") from e
