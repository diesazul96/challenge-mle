import unittest
from unittest.mock import ANY, MagicMock, patch

from fastapi.testclient import TestClient
from mockito import when, unstub
import numpy as np
from challenge import app


class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.mock_model_instance = MagicMock()

        self.model_patcher = patch("challenge.api.model", self.mock_model_instance)
        self.model_patcher.start()

        self.client = TestClient(app)

    def tearDown(self):
        self.model_patcher.stop()
        unstub()

    def test_should_get_predict(self):
        self.mock_model_instance._trained_airlines = {"Aerolineas Argentinas"}
        self.mock_model_instance.predict.return_value = np.array([0])

        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}]
        }
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})

    def test_should_failed_unkown_column_1(self):
        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 13}]
        }
        when("sklearn.linear_model.LogisticRegression").predict(ANY).thenReturn(
            np.array([0])
        )
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_2(self):
        data = {
            "flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "O", "MES": 13}]
        }
        when("sklearn.linear_model.LogisticRegression").predict(ANY).thenReturn(
            np.array([0])
        )
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unkown_column_3(self):
        data = {"flights": [{"OPERA": "Argentinas", "TIPOVUELO": "O", "MES": 13}]}
        when("sklearn.linear_model.LogisticRegression").predict(ANY).thenReturn(
            np.array([0])
        )
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)
