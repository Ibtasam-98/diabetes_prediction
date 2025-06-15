from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

app = Flask(__name__)

# Load the dataset and preprocess it
data = pd.read_csv("diabetes.csv")
X = data[['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']]
y = data["Outcome"]

# Feature scaling and polynomial feature addition
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Hyperparameter tuning and model training
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Save the model and scaler using joblib
joblib.dump(grid_search.best_estimator_, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly.pkl')

# Load the model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
poly = joblib.load('poly.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    glucose = data['Glucose']
    blood_pressure = data['BloodPressure']
    insulin = data['Insulin']
    bmi = data['BMI']
    age = data['Age']

    # Prepare the input data
    user_data = pd.DataFrame([[glucose, blood_pressure, insulin, bmi, age]],
                             columns=['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age'])

    # Scale and transform the input data
    user_data_scaled = scaler.transform(user_data)
    user_data_poly = poly.transform(user_data_scaled)

    # Make prediction
    prediction = model.predict(user_data_poly)

    # Return prediction result
    result = {
        'prediction': int(prediction[0]),  # Convert to int for JSON response
        'prediction_label': 'Diabetes' if prediction[0] == 1 else 'No Diabetes'
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
