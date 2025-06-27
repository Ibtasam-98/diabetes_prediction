from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  # Changed from RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Load and preprocess data
try:
    print("Loading dataset...")
    data = pd.read_csv("dataset/diabetes.csv")
    print("Dataset loaded successfully")
except FileNotFoundError:
    print("Error: diabetes.csv not found. Please ensure the dataset is in the same directory.")
    exit()

# Define features and target
features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']
target = 'Outcome'

X = data[features]
y = data[target]

# Preprocessing
print("Preprocessing data...")
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
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced', None]
        }
    },
    'KNN': {
        'model': None,
        'accuracy': None,
        'params': {
            'n_neighbors': range(1, 21),
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }
    },
    'Decision Tree': {
        'model': None,
        'accuracy': None,
        'params': {
            'max_depth': [None] + list(range(3, 21)),
            'min_samples_split': range(2, 11),
            'min_samples_leaf': range(1, 11),
            'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2', None]
        }
    },
    'SVM': {  # Changed from Random Forest to SVM
        'model': None,
        'accuracy': None,
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['linear', 'rbf', 'poly'],
            'class_weight': ['balanced', None],
            'probability': [True]  # Required for predict_proba
        }
    }
}

# Train and evaluate models
print("\nTraining models...")
for model_name, model_data in models.items():
    print(f"\nTraining {model_name}...")

    if model_name == 'Logistic Regression':
        grid = GridSearchCV(LogisticRegression(max_iter=1000),
                          model_data['params'],
                          cv=5,
                          scoring='accuracy',
                          n_jobs=-1,
                          error_score='raise')
    elif model_name == 'KNN':
        grid = GridSearchCV(KNeighborsClassifier(),
                          model_data['params'],
                          cv=5,
                          scoring='accuracy',
                          n_jobs=-1,
                          error_score='raise')
    elif model_name == 'Decision Tree':
        grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                          model_data['params'],
                          cv=5,
                          scoring='accuracy',
                          n_jobs=-1,
                          error_score='raise')
    else:  # SVM
        grid = GridSearchCV(SVC(random_state=42),
                          model_data['params'],
                          cv=5,
                          scoring='accuracy',
                          n_jobs=-1,
                          error_score='raise')

    grid.fit(X_train, y_train)
    models[model_name]['model'] = grid.best_estimator_
    models[model_name]['accuracy'] = accuracy_score(y_test, grid.best_estimator_.predict(X_test))
    print(f"{model_name} trained with accuracy: {models[model_name]['accuracy']:.4f}")
    print(f"Best parameters: {grid.best_params_}")

# Save models and preprocessing objects
print("\nSaving models...")
joblib.dump(models, 'models.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(poly, 'poly.pkl')
print("Models saved successfully")


@app.route('/predict', methods=['POST'])
def predict():
    print("\nReceived prediction request...")
    try:
        # Get input data
        input_data = request.json
        print("Input data:", input_data)

        # Validate input
        required_fields = features
        for field in required_fields:
            if field not in input_data:
                print(f"Missing field: {field}")
                return jsonify({'error': f'Missing required field: {field}'}), 400
            if not isinstance(input_data[field], (int, float)):
                print(f"Invalid value for {field}: {input_data[field]}")
                return jsonify({'error': f'Invalid value for {field}. Must be numeric.'}), 400

        # Prepare input for prediction
        user_data = pd.DataFrame([[input_data[field] for field in features]],
                               columns=features)
        print("Input DataFrame:\n", user_data)

        # Preprocess input
        user_data_scaled = scaler.transform(user_data)
        user_data_poly = poly.transform(user_data_scaled)

        # Make predictions with all models
        predictions = {}
        for model_name, model_data in models.items():
            print(f"\nMaking prediction with {model_name}...")
            pred = model_data['model'].predict(user_data_poly)[0]
            proba = model_data['model'].predict_proba(user_data_poly)[0][1]
            print(f"{model_name} prediction: {'Diabetes' if pred == 1 else 'No Diabetes'} (probability: {proba:.4f})")

            predictions[model_name] = {
                'prediction': 'Diabetes' if pred == 1 else 'No Diabetes',
                'probability': float(proba),
                'model_accuracy': float(model_data['accuracy'])
            }

        # Determine consensus prediction
        diabetes_votes = sum(1 for pred in predictions.values() if pred['prediction'] == 'Diabetes')
        no_diabetes_votes = len(predictions) - diabetes_votes
        print(f"\nVotes - Diabetes: {diabetes_votes}, No Diabetes: {no_diabetes_votes}")

        if diabetes_votes > no_diabetes_votes:
            consensus = 'Diabetes'
        elif no_diabetes_votes > diabetes_votes:
            consensus = 'No Diabetes'
        else:
            # If tie, use model with higher accuracy
            best_model = max(predictions.items(), key=lambda x: x[1]['model_accuracy'])
            consensus = best_model[1]['prediction']
            print(f"Tie broken using {best_model[0]} (accuracy: {best_model[1]['model_accuracy']:.4f})")

        print(f"Consensus prediction: {consensus}")

        # Prepare response
        response = {
            'consensus_prediction': consensus,
            'model_predictions': predictions,
            'status': 'success'
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    print("\nReceived model info request")
    try:
        model_info = {}
        for model_name, model_data in models.items():
            model_info[model_name] = {
                'accuracy': float(model_data['accuracy']),
                'parameters': str(model_data['model'].get_params())
            }
        print("Returning model info")
        return jsonify({'models': model_info, 'status': 'success'})
    except Exception as e:
        print(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e), 'status': 'error'}), 500


if __name__ == '__main__':
    print("\nStarting Flask server...")
    app.run(debug=True, port=5001)