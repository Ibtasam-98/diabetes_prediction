from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Load and preprocess data
try:
    data = pd.read_csv("dataset/diabetes.csv")
except FileNotFoundError:
    print("Error: diabetes.csv not found. Please ensure the dataset is in the same directory.")
    exit()

# Define features and target
features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
target = 'Outcome'

X = data[features]
y = data[target]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Initialize models dictionary to store all models and their accuracies
models = {
    'Logistic Regression': {
        'model': None,
        'accuracy': None,
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['liblinear'],
            'class_weight': ['balanced', None]
        }
    },
    'KNN': {
        'model': None,
        'accuracy': None,
        'params': {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }
}

# Train and evaluate models
for model_name, model_data in models.items():
    if model_name == 'Logistic Regression':
        grid = GridSearchCV(LogisticRegression(max_iter=500),
                            model_data['params'],
                            cv=5,
                            scoring='accuracy')
    else:  # KNN
        grid = GridSearchCV(KNeighborsClassifier(),
                            model_data['params'],
                            cv=5,
                            scoring='accuracy')

    grid.fit(X_train, y_train)
    models[model_name]['model'] = grid.best_estimator_
    models[model_name]['accuracy'] = accuracy_score(y_test, grid.best_estimator_.predict(X_test))

# Save models and preprocessing objects
joblib.dump(models, 'models.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = request.json

        # Validate input
        required_fields = features
        for field in required_fields:
            if field not in input_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            if not isinstance(input_data[field], (int, float)):
                return jsonify({'error': f'Invalid value for {field}. Must be numeric.'}), 400

        # Prepare input for prediction
        user_data = pd.DataFrame([[input_data[field] for field in features]],
                                 columns=features)

        # Preprocess input
        user_data_scaled = scaler.transform(user_data)
        user_data_poly = poly.transform(user_data_scaled)

        # Make predictions with all models
        predictions = {}
        for model_name, model_data in models.items():
            pred = model_data['model'].predict(user_data_poly)[0]
            predictions[model_name] = {
                'prediction': 'Diabetes' if pred == 1 else 'No Diabetes',
                'probability': float(model_data['model'].predict_proba(user_data_poly)[0][1]),
                'model_accuracy': float(model_data['accuracy'])
            }

        # Determine consensus prediction
        diabetes_votes = sum(1 for pred in predictions.values() if pred['prediction'] == 'Diabetes')
        no_diabetes_votes = len(predictions) - diabetes_votes

        if diabetes_votes > no_diabetes_votes:
            consensus = 'Diabetes'
        elif no_diabetes_votes > diabetes_votes:
            consensus = 'No Diabetes'
        else:
            # If tie, use model with higher accuracy
            best_model = max(predictions.items(), key=lambda x: x[1]['model_accuracy'])
            consensus = best_model[1]['prediction']

        # Prepare response
        response = {
            'consensus_prediction': consensus,
            'model_predictions': predictions,
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        model_info = {}
        for model_name, model_data in models.items():
            model_info[model_name] = {
                'accuracy': float(model_data['accuracy']),
                'parameters': str(model_data['model'].get_params())
            }
        return jsonify({'models': model_info, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)