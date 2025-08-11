# SmartGluco: A Mobile Health Solution for Diabetes Risk Assessment

---

SmartGluco is an end to end mobile health system designed to provide real-time diabetes risk assessment using machine learning. This repository contains the backend API, the machine learning model training script, and the necessary data and saved model artifacts.

## Features

* **Optimized Machine Learning Model:** A Logistic Regression model enhanced with second-degree polynomial features for improved predictive power, achieving a **73% accuracy** on the test set.
* **Scalable RESTful API:** A lightweight Flask backend to serve real-time diabetes risk predictions with sub-500ms response times.
* **Intuitive Mobile Application Integration (Frontend not included in this repo):** Designed to connect with a Flutter mobile application (as detailed in the accompanying [research paper](https://github.com/Ibtasam-98/smartgluco/blob/main/smartGluco.pdf)) for interactive data input and clear, color-coded results.
* **Automated Model Training & Persistence:** Includes a script (`main.py`) to train the model and save the trained model, scaler, and polynomial feature transformer for easy deployment.

## Getting Started

Follow these steps to get a copy of the project up and running on your local machine.

### Data Setup

* Ensure the `diabetes.csv` dataset is present in the root directory of the project. This dataset is crucial for both model training and the functioning of the API.

### Running the Backend API

The `app.py` script serves the Flask REST API for predictions.

1.  **Ensure model artifacts exist:** Run `main.py` first (see next section) to generate `model.pkl`, `scaler.pkl`, and `poly.pkl` if they are not already present.
2.  **Start the Flask development server:**
    ```bash
    python app.py
    ```
    The API will typically run on `http://127.0.0.1:5001`.

### Running the Model Training and Analysis

The `main.py` script handles data loading, preprocessing, model training, hyperparameter tuning, and evaluation, including generating plots.

1.  **Execute the script:**
    ```bash
    python main.py
    ```
    This script will:
    * Load the `diabetes.csv` dataset.
    * Perform Exploratory Data Analysis (EDA) and display plots (pairplot, correlation matrix).
    * Preprocess the data (scaling, polynomial features).
    * Train a Logistic Regression model with `GridSearchCV` for hyperparameter tuning.
    * Print the best model parameters and cross-validation accuracy.
    * Generate and display the classification report, confusion matrices, and ROC curve for the test set.
    * Save the trained `model.pkl`, `scaler.pkl`, and `poly.pkl` files to the project directory.
    * Prompt for manual patient input to demonstrate a prediction.

## 🤖 API Endpoints

The Flask API provides the following endpoints:

### 1. Predict Diabetes Risk (`/predict`)

* **Method:** `POST`
* **Description:** Accepts patient health parameters and returns a diabetes risk prediction.
* **Request Body (JSON):**
    ```json
    {
        "Glucose": 148,
        "BloodPressure": 72,
        "Insulin": 0,
        "BMI": 33.6,
        "Age": 50
    }
    ```
* **Response (JSON):**
    ```json
    {
        "prediction": 1,
        "prediction_label": "Diabetes"
    }
    ```
    (or `"prediction": 0, "prediction_label": "No Diabetes"`)

### 2. Get Model Accuracy (`/accuracy`)

* **Method:** `GET`
* **Description:** Returns the last calculated accuracy of the model on the test set.
* **Response (JSON):**
    ```json
    {
        "model_accuracy": 0.7338
    }
    ```

## 📂 Project Structure
