# Challenge Solution Documentation

## 1. Project Overview

This project aims to operationalize a flight delay prediction model for SCL airport. The original work, provided as a Jupyter Notebook (`exploration.ipynb`), involved training a model to predict the probability of flight delays. This solution transcribes the model into a production-ready Python script, exposes it via a FastAPI, deploys it to Google Cloud Platform (GCP), and establishes a CI/CD pipeline for automated testing and deployment.

The primary goal is to provide a reliable API endpoint that the airport team can query to get delay predictions for specific flights.

## 2. Design Decisions

### 2.1. Model Implementation (`challenge/model.py`)

*   **Model Choice:** The model implemented is a Logistic Regression. This was chosen based on the DS's work in the notebook, since there's no clear difference in performance between XGBoost and LogisticRegression, and since this model has lower complexity and in general faster predictions is a reliable model to use on an API. 
*   **Preprocessing Strategy:**
    *   The `preprocess` method handles the transformation of raw input data (either from the training dataset or an API request) into a format suitable for the Logistic Regression model.
    *   **Feature Engineering (Training Path):** When a `target_column` is provided (i.e., during training or initial data processing), the method calculates `min_diff` (difference between scheduled and operational flight times) and derives the binary `delay` target variable based on a 15-minute threshold. It also captures the set of unique airline names (`_trained_airlines`) from the `OPERA` column for later validation during prediction.
    *   **Feature Engineering (Prediction Path):** For API predictions, the `min_diff` and target creation steps are skipped.
    *   **One-Hot Encoding:** `pd.get_dummies` is used to convert categorical features (`OPERA`, `TIPOVUELO`, `MES`) into a numerical format.
    *   **Consistent Feature Set (`FEATURES_COLS`):** A predefined list (based on the DS analysis), `FEATURES_COLS`, ensures that the model always receives the same 10 input features in the correct order. If a category present in an input request wasn't seen during training or doesn't result in one of these 10 features directly, it's handled by ensuring all columns in `FEATURES_COLS` exist, defaulting to 0 if a feature wasn't generated from the input. This makes the prediction process robust to new or rare categorical values not explicitly among the top features.
*   **Fitting (`fit` method):**
    *   The model calculates class weights to handle potential class imbalance.
    *   The `LogisticRegression` model is initialized with these class weights.
*   **Prediction (`predict` method):**
    *   Takes the preprocessed features and returns a list of predictions.
    *   Includes a `NotFittedError` check to ensure the model has been trained.

### 2.2. API Design (`challenge/api.py`)

*   **Framework:** FastAPI was used as required by the challenge.
*   **Model Loading:** The trained model is loaded once at application startup (`@app.on_event("startup")`) from a path specified in `challenge/config.py` (`API_MODEL_PATH`). This avoids reloading the model on every request, improving performance. Robust error handling is included in case the model file is not found or fails to load, raising an HTTP 503 error.
*   **Input Validation (`challenge/schemas.py`):**
    *   Pydantic models (`PredictionRequest`, `PredictionResponse`) are used for request and response validation and serialization.
    *   `PredictionRequest` uses `typing.Literal` to strictly enforce allowed values for `TIPOVUELO` (I, N) and `MES` (1-12), ensuring data integrity at the API.
*   **Endpoints:**
    *   `/health`: A simple health check endpoint returning `{"status": "OK"}`.
    *   `/predict` (POST):
        *   Accepts a JSON payload containing flight details.
        *   Validates the input using `PredictionRequest`.
        *   Performs an additional validation check to ensure the provided `OPERA` (airline) was seen during model training (using `model._trained_airlines`). This prevents attempts to predict on airlines the model knows nothing about, which could lead to unreliable predictions.
        *   Calls `model.preprocess()` and `model.predict()`.
        *   Returns predictions in the `PredictionResponse` format.
*   **Error Handling:**
    *   Specific `HTTPException`s are raised for different error scenarios:
        *   `400 (Bad Request)`: For invalid input data (e.g., `ValueError` from model validation like an unseen airline).
        *   `500 (Internal Server Error)`: For unexpected errors during the prediction process.
        *   `503 (Service Unavailable)`: If the model fails to load at startup.
    *   Logging is implemented to record errors and key events.

### 2.3. Training Pipeline (`challenge/train_pipeline.py`)

*   **Purpose:** This script orchestrates the model training process.
*   **Steps:**
    1.  Loads the raw data (`data/data.csv`).
    2.  Initializes the `DelayModel`.
    3.  Preprocesses the data using `model.preprocess()` to get features (`x`) and the target variable (`y`).
    4.  Splits the data into training and validation sets using `train_test_split` (with `VALIDATION_RATIO` and `RANDOM_STATE` from `config.py`).
    5.  Fits the model on the training set (`model.fit(x_train, y_train)`) and makes predictions on the validation set.
    6.  Generates and saves a classification report (`metrics/classification_report_....json`).
    7.  **Retrains the model on the full dataset (`x`, `y`)**. To create the final production model using all available data.
    8.  Saves the final trained model locally using `joblib.dump()` to `MODEL_OUTPUT_PATH` (which includes a timestamp).

### 2.4. Configuration (`challenge/config.py`)

*   Centralizes key paths, column names, and parameters.

### 2.5. GCP Deployment

*   **Cloud Run:** The FastAPI application is containerized (using a `Dockerfile`) and deployed as a serverless service on Google Cloud Run. This provides scalability and integrates well with other GCP services.
*   **Artifact Registry:** Docker images built by the CI/CD pipeline are pushed to Google Artifact Registry, which serves as a private Docker registry.
*   **Workload Identity Federation:** Secure authentication between GitHub Actions and GCP is established using Workload Identity Federation. This allows GitHub Actions to impersonate a GCP service account (`github-actions-deployer@challenge-mle-dam.iam.gserviceaccount.com`) without needing to manage long-lived service account keys. The service account is granted necessary permissions (e.g., to push to Artifact Registry, deploy to Cloud Run).

### 2.6. CI/CD Pipeline (`.github/workflows/`)

*   **Technology:** GitHub Actions.
*   **`ci.yml` (Continuous Integration):**
    *   **Triggers:** Runs on pushes and pull requests to `master` and `develop` branches.
    *   **Jobs:**
        *   `ci`:
            1.  Checks out code.
            2.  Sets up Python and Poetry.
            3.  Caches Poetry virtual environment for speed.
            4.  Installs dependencies using Poetry.
            5.  Lints code using Ruff (for checking and formatting).
            6.  Runs tests using Pytest.
            7.  Performs a security scan using Bandit.
            8.  Tests if the Docker image can be built successfully.
*   **`cd.yml` (Continuous Delivery):**
    *   **Triggers:**
        *   `workflow_run`: When the "Continuous Integration" workflow completes successfully (for `develop` and `master` branches).
        *   `push`: Directly on pushes to `master` and `develop`.
        *   `workflow_dispatch`: Allows manual triggering for a specified branch. (For debugging and test)
    *   **Jobs:**
        *   `cd-staging`:
            *   **Condition:** Runs if CI succeeded on `develop`, or on a direct push to `develop`, or if manually dispatched.
            *   **Steps:**
                1.  Checks out code.
                2.  Authenticates to Google Cloud using Workload Identity Federation.
                3.  Sets up Google Cloud SDK.
                4.  Configures Docker for Artifact Registry.
                5.  Builds and pushes a Docker image tagged with `develop-<sha>` to Artifact Registry.
                6.  Deploys this image to a Cloud Run service suffixed with `-staging`.
                7.  Captures the staging URL and runs a smoke test (curls the `/health` endpoint).
        *   `cd-production`:
            *   **Condition:** Runs if CI succeeded on `master` , or on a direct push to `master`.
            *   **Steps:** (Same as staging but targets the production environment)

## 3. Possible Improvements

*   **`challenge/model.py`:** Avoid `pd.get_dummies` execution mutiple times by defining a template `DataFrame` with the know expected columns.
*   **`challenge/train_pipeline.py`:** Save artifacts (model and metrics) to an artifact storage like a bucket in order to used it as a repository and avoid storage of those on the repository.
*   **`challenge/api.py`:** Load model from a defined storage (bucket).
*   **`.github/workflows/*.yml`:** Change step in a single job to multiple jobs with unique responsability for better resource utilization and guarantee that if one of the jobs fails the other ones won't be executed
