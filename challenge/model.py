from datetime import datetime
import logging
from typing import Tuple, Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
THRESHOLD_IN_MINUTES = 15
RANDOM_STATE = 42

ORIGINAL_COLS = [
    "OPERA",
    "TIPOVUELO",
    "MES",   
]

FEATURES_COLS = [
    "OPERA_Latin American Wings", 
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air"
]

TARGET_COL = ["delay"]

class DelayModel:
    """
    A machine learning model for predicting flight delays.
    
    This class implements a logistic regression model to predict whether a flight
    will be delayed based on various features such as airline, flight type, and month.
    
    Attributes:
        _model (LogisticRegression): The trained logistic regression model.
    """
    
    def __init__(self) -> None:
        """Initialize the DelayModel with no trained model."""
        self._model: Optional[LogisticRegression] = None

    @staticmethod
    def _get_min_diff(data: pd.DataFrame) -> float:
        """
        Calculate difference in minutes between Fecha-O and Fecha-I.

        Args:
            data (pd.DataFrame): Raw data containing Fecha-O and Fecha-I columns.

        Returns:
            float: Difference in minutes between Fecha-O and Fecha-I.
            
        Raises:
            ValueError: If required columns are missing or date format is invalid.
        """
        try:
            fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
            fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
            return ((fecha_o - fecha_i).total_seconds()) / 60
        except KeyError as e:
            raise ValueError(f"Missing required column: {e}") from e
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}") from e

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): Raw data.
            target_column (str, optional): If set, the target is returned.

        Returns:
            Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]: 
                - If target_column is provided: Tuple of (features, target)
                - Otherwise: features DataFrame

        Raises:
            ValueError: If required columns are missing.
        """
        missing_cols = [col for col in ORIGINAL_COLS if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        target = None
        if target_column:
            data['min_diff'] = data.apply(self._get_min_diff, axis=1)
            data[target_column] = np.where(data['min_diff'] > THRESHOLD_IN_MINUTES, 1, 0)
            target = data[[target_column]]
            
        # Optimize preprocessing by using categorical dtype
        features = data[ORIGINAL_COLS].astype({
            'OPERA': 'category',
            'TIPOVUELO': 'category',
            'MES': 'category'
        })
        
        # Create dummy variables more efficiently
        features = pd.get_dummies(features, prefix=['OPERA', 'TIPOVUELO', 'MES'])

        features = features[FEATURES_COLS]
        logger.info(f"Preprocessed data shape: {features.shape}")

        return features if target is None else (features, target)

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): Preprocessed features.
            target (pd.DataFrame): Target values.

        Raises:
            ValueError: If input data is invalid.
        """
        if features.empty or target.empty:
            raise ValueError("Empty features or target data provided")

        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])

        self._model = LogisticRegression(
            class_weight={1: n_y0/len(target), 0: n_y1/len(target)},
            random_state=RANDOM_STATE
        )
        self._model.fit(features, target.values.ravel())
        logger.info("Model training completed successfully")

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): Preprocessed features.

        Returns:
            List[int]: Predicted targets.

        Raises:
            NotFittedError: If the model hasn't been trained.
            ValueError: If input data is invalid.
        """
        if self._model is None:
            raise NotFittedError("Model has not been trained yet")
            
        if features.empty:
            raise ValueError("Empty features data provided")
            
        return self._model.predict(features).tolist()
