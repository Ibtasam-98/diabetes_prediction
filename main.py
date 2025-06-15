import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Load the datase
data = pd.read_csv("diabetes.csv")
print("Dataset Shape:", data.shape)

# Check for missing values
print("Missing values in each column:")
print(data.isnull().sum())

# Data Visualization
# Visualize only the selected features and Outcome
selected_features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'Outcome']
sns.pairplot(data[selected_features], hue='Outcome', height=2.5)
plt.suptitle('Pairplot of Selected Features', y=1.02)
plt.show()

# Correlation matrix for selected features
plt.figure(figsize=(10, 8))  # Adjust figure size for better visibility
sns.heatmap(data[selected_features].corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Selected Features', fontsize=16)
plt.show()

# Select important features
data = data[selected_features]

# Define features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
print("Original number of features:", X.shape[1])
print("Number of features after adding polynomial features:", X_poly.shape[1])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Hyperparameter tuning for Logistic Regression
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['liblinear'],
    'class_weight': ['balanced', None]
}
grid_search = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and model evaluation
best_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_)

# Predictions and evaluation metrics
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# Classification report for Testing Data
print("\n--- Classification Report (Testing Data) ---")
print(classification_report(y_test, y_test_pred))

# Confusion Matrix
def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'])
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)
print("\n--- Confusion Matrix - Training Data ---")
print(cm_train)
plot_confusion_matrix(cm_train, title='Training Data Confusion Matrix')
print("\n--- Confusion Matrix - Testing Data ---")
print(cm_test)
plot_confusion_matrix(cm_test, title='Testing Data Confusion Matrix')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# User input for prediction
print("\n--- Enter Patient Information for Prediction ---")
glucose = float(input("Enter Glucose: "))
blood_pressure = float(input("Enter BloodPressure: "))
insulin = float(input("Enter Insulin: "))
bmi = float(input("Enter BMI: "))
age = float(input("Enter Age: "))

user_data_df = pd.DataFrame([[glucose, blood_pressure, insulin, bmi, age]], columns=['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age'])
user_data_scaled = scaler.transform(user_data_df)
user_data_poly = poly.transform(user_data_scaled)
prediction = best_model.predict(user_data_poly)

# Display prediction
if prediction[0] == 1:
    print("Prediction: The person is likely to have diabetes.")
else:
    print("Prediction: The person is likely not to have diabetes.")

# Print the testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
# print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")
print("Model evaluation and visualization completed.")
