from pydantic import BaseModel
from typing import Literal

ALLOWED_TIPOVUELO = Literal["I", "N"]
ALLOWED_MES = Literal[
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
]

class PredictionRequest(BaseModel):
    OPERA: str
    TIPOVUELO: ALLOWED_TIPOVUELO
    MES: ALLOWED_MES


class PredictionResponse(BaseModel):
    predict: list[int]
