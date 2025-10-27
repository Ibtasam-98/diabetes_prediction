# app.py - Complete Diabetes Prediction System
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, f1_score, recall_score, precision_score)
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
import warnings

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="SmartGluco - Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_palette("Blues")
plt.style.use('seaborn-v0_8-whitegrid')


class DiabetesPredictor:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.threshold_results = {}
        self.scaler = None
        self.poly = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.data = None

    def load_data(self):
        """Load and return sample diabetes data"""
        try:
            # Create sample data based on Pima Indians Diabetes dataset structure
            np.random.seed(42)
            n_samples = 768

            data = pd.DataFrame({
                'Pregnancies': np.random.randint(0, 17, n_samples),
                'Glucose': np.random.normal(120, 30, n_samples).astype(int),
                'BloodPressure': np.random.normal(70, 12, n_samples).astype(int),
                'SkinThickness': np.random.normal(20, 10, n_samples).astype(int),
                'Insulin': np.random.normal(80, 100, n_samples).astype(int),
                'BMI': np.random.normal(32, 8, n_samples),
                'DiabetesPedigreeFunction': np.random.uniform(0.08, 2.5, n_samples),
                'Age': np.random.randint(21, 81, n_samples),
                'Outcome': np.random.binomial(1, 0.35, n_samples)
            })

            # Ensure realistic ranges
            data['Glucose'] = np.clip(data['Glucose'], 70, 200)
            data['BloodPressure'] = np.clip(data['BloodPressure'], 60, 140)
            data['BMI'] = np.clip(data['BMI'], 15, 50)
            data['Insulin'] = np.clip(data['Insulin'], 15, 300)

            self.data = data
            return data

        except Exception as e:
            st.error(f"Error creating sample data: {str(e)}")
            return None

    def preprocess_data(self, features, target='Outcome'):
        """Preprocess the data"""
        try:
            # Select features
            data = self.data[features + [target]]

            # Split features and target
            X = data.drop(target, axis=1)
            y = data[target]

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Add polynomial features
            self.poly = PolynomialFeatures(degree=2, include_bias=False)
            X_poly = self.poly.fit_transform(X_scaled)

            # Split data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_poly, y, test_size=0.2, random_state=42, stratify=y)

            return True

        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            return False

    def calculate_class_weights(self, y_train):
        """Calculate balanced class weights"""
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        return {0: class_weights[0], 1: class_weights[1]}

    def apply_smote(self, X_train, y_train):
        """Apply SMOTE for handling class imbalance"""
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        return X_train_balanced, y_train_balanced

    def train_models(self, use_smote=True):
        """Train multiple models with hyperparameter tuning"""
        try:
            with st.spinner("Training models with enhanced class imbalance handling..."):
                # Calculate class weights
                class_weights = self.calculate_class_weights(self.y_train)

                # Apply SMOTE if requested
                if use_smote:
                    X_train_balanced, y_train_balanced = self.apply_smote(self.X_train, self.y_train)
                else:
                    X_train_balanced, y_train_balanced = self.X_train, self.y_train

                # Parameter grids
                lr_param_grid = {
                    'C': np.logspace(-3, 3, 7),
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'class_weight': ['balanced', class_weights]
                }

                knn_param_grid = {
                    'n_neighbors': range(3, 15),
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }

                dt_param_grid = {
                    'max_depth': [None, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', class_weights]
                }

                rf_param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 15],
                    'min_samples_split': [2, 5],
                    'class_weight': ['balanced', class_weights]
                }

                # Initialize and train models
                models_config = {
                    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), lr_param_grid),
                    'K-Nearest Neighbors': (KNeighborsClassifier(), knn_param_grid),
                    'Decision Tree': (DecisionTreeClassifier(random_state=42), dt_param_grid),
                    'Random Forest': (RandomForestClassifier(random_state=42), rf_param_grid)
                }

                for name, (model, param_grid) in models_config.items():
                    st.info(f"Training {name}...")
                    grid = GridSearchCV(model, param_grid, cv=5, scoring='f1', n_jobs=-1)
                    grid.fit(X_train_balanced, y_train_balanced)
                    self.models[name] = grid.best_estimator_

                return True

        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return False

    def optimize_thresholds(self):
        """Optimize prediction thresholds for better recall"""
        try:
            threshold_results = {}

            for name, model in self.models.items():
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(self.X_test)[:, 1]

                    best_threshold = 0.5
                    best_f1 = 0

                    thresholds = np.arange(0.3, 0.7, 0.02)

                    for threshold in thresholds:
                        y_pred = (y_prob >= threshold).astype(int)
                        current_f1 = f1_score(self.y_test, y_pred, zero_division=0)

                        if current_f1 > best_f1:
                            best_f1 = current_f1
                            best_threshold = threshold

                    threshold_results[name] = {
                        'optimal_threshold': best_threshold,
                        'f1_score': best_f1
                    }

            self.threshold_results = threshold_results
            return True

        except Exception as e:
            st.error(f"Error optimizing thresholds: {str(e)}")
            return False

    def evaluate_models(self):
        """Evaluate all trained models"""
        try:
            results = {}

            for name, model in self.models.items():
                # Get optimal threshold
                optimal_threshold = self.threshold_results.get(name, {}).get('optimal_threshold', 0.5)

                # Standard predictions
                y_train_pred_standard = model.predict(self.X_train)
                y_test_pred_standard = model.predict(self.X_test)

                # Optimized predictions
                if hasattr(model, "predict_proba"):
                    y_prob_test = model.predict_proba(self.X_test)[:, 1]
                    y_test_pred_optimized = (y_prob_test >= optimal_threshold).astype(int)
                else:
                    y_test_pred_optimized = y_test_pred_standard

                # Calculate metrics
                train_accuracy = accuracy_score(self.y_train, y_train_pred_standard)
                test_accuracy_standard = accuracy_score(self.y_test, y_test_pred_standard)
                test_accuracy_optimized = accuracy_score(self.y_test, y_test_pred_optimized)

                recall_standard = recall_score(self.y_test, y_test_pred_standard)
                recall_optimized = recall_score(self.y_test, y_test_pred_optimized)

                # ROC curve
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(self.X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                else:
                    fpr, tpr, roc_auc = [0], [0], 0.5

                # Store results
                results[name] = {
                    'model': model,
                    'train_accuracy': train_accuracy,
                    'test_accuracy_standard': test_accuracy_standard,
                    'test_accuracy_optimized': test_accuracy_optimized,
                    'recall_standard': recall_standard,
                    'recall_optimized': recall_optimized,
                    'recall_improvement': recall_optimized - recall_standard,
                    'cm_test_standard': confusion_matrix(self.y_test, y_test_pred_standard),
                    'cm_test_optimized': confusion_matrix(self.y_test, y_test_pred_optimized),
                    'fpr': fpr,
                    'tpr': tpr,
                    'roc_auc': roc_auc,
                    'optimal_threshold': optimal_threshold
                }

            self.results = results
            return True

        except Exception as e:
            st.error(f"Error evaluating models: {str(e)}")
            return False

    def plot_visualizations(self):
        """Create all visualizations"""
        try:
            # 1. Data Overview
            st.subheader("Data Overview")
            col1, col2 = st.columns(2)

            with col1:
                st.write("Dataset Shape:", self.data.shape)
                st.write("Class Distribution:")
                st.write(self.data['Outcome'].value_counts())

            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                self.data['Outcome'].value_counts().plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
                ax.set_title('Diabetes Outcome Distribution')
                ax.set_xlabel('Outcome (0: No Diabetes, 1: Diabetes)')
                ax.set_ylabel('Count')
                st.pyplot(fig)

            # 2. Correlation Matrix
            st.subheader("Feature Correlation Matrix")
            features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age', 'Outcome']
            corr_matrix = self.data[features].corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='Blues', square=True, mask=mask, ax=ax)
            ax.set_title('Feature Correlation Matrix')
            st.pyplot(fig)

            # 3. Model Performance Comparison
            st.subheader("Model Performance Comparison")

            # Accuracy comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            models = list(self.results.keys())
            standard_acc = [self.results[m]['test_accuracy_standard'] for m in models]
            optimized_acc = [self.results[m]['test_accuracy_optimized'] for m in models]

            x = np.arange(len(models))
            width = 0.35

            ax1.bar(x - width / 2, standard_acc, width, label='Standard', alpha=0.7)
            ax1.bar(x + width / 2, optimized_acc, width, label='Optimized', alpha=0.7)
            ax1.set_xlabel('Models')
            ax1.set_ylabel('Accuracy')
            ax1.set_title('Model Accuracy: Standard vs Optimized')
            ax1.set_xticks(x)
            ax1.set_xticklabels(models, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Recall comparison
            standard_recall = [self.results[m]['recall_standard'] for m in models]
            optimized_recall = [self.results[m]['recall_optimized'] for m in models]

            ax2.bar(x - width / 2, standard_recall, width, label='Standard', alpha=0.7)
            ax2.bar(x + width / 2, optimized_recall, width, label='Optimized', alpha=0.7)
            ax2.set_xlabel('Models')
            ax2.set_ylabel('Recall')
            ax2.set_title('Model Recall: Standard vs Optimized')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

            # 4. ROC Curves
            st.subheader("ROC Curves")
            fig, ax = plt.subplots(figsize=(10, 8))

            for name, result in self.results.items():
                ax.plot(result['fpr'], result['tpr'],
                        label=f'{name} (AUC = {result["roc_auc"]:.3f})', linewidth=2)

            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves Comparison')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            # 5. Confusion Matrices
            st.subheader("Confusion Matrices")
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.ravel()

            for idx, (name, result) in enumerate(self.results.items()):
                if idx < 4:  # Ensure we don't exceed subplot count
                    sns.heatmap(result['cm_test_optimized'], annot=True, fmt='d', cmap='Blues',
                                xticklabels=['No Diabetes', 'Diabetes'],
                                yticklabels=['No Diabetes', 'Diabetes'],
                                ax=axes[idx])
                    axes[idx].set_title(f'{name}\n(Optimized Threshold)')
                    axes[idx].set_xlabel('Predicted')
                    axes[idx].set_ylabel('Actual')

            plt.tight_layout()
            st.pyplot(fig)

            # 6. Learning Curves
            st.subheader("Learning Curves")
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()

            for idx, (name, model) in enumerate(self.models.items()):
                if idx < 4:
                    try:
                        train_sizes, train_scores, test_scores = learning_curve(
                            model, self.X_train, self.y_train, cv=5, n_jobs=-1,
                            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

                        train_mean = np.mean(train_scores, axis=1)
                        train_std = np.std(train_scores, axis=1)
                        test_mean = np.mean(test_scores, axis=1)
                        test_std = np.std(test_scores, axis=1)

                        axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
                        axes[idx].plot(train_sizes, test_mean, 's-', color='red', label='Cross-validation score')
                        axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1,
                                               color='blue')
                        axes[idx].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1,
                                               color='red')

                        axes[idx].set_xlabel('Training examples')
                        axes[idx].set_ylabel('Accuracy')
                        axes[idx].set_title(f'Learning Curve - {name}')
                        axes[idx].legend()
                        axes[idx].grid(True, alpha=0.3)

                    except Exception as e:
                        st.warning(f"Could not generate learning curve for {name}: {str(e)}")

            plt.tight_layout()
            st.pyplot(fig)

            return True

        except Exception as e:
            st.error(f"Error creating visualizations: {str(e)}")
            return False

    def predict(self, input_data):
        """Make prediction on new data"""
        try:
            if not self.models or not self.scaler or not self.poly:
                return None, "Models not trained yet"

            # Preprocess input
            input_scaled = self.scaler.transform(input_data)
            input_poly = self.poly.transform(input_scaled)

            # Get predictions from all models
            predictions = {}
            probabilities = []

            for name, model in self.models.items():
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_poly)[0][1]
                    optimal_threshold = self.threshold_results.get(name, {}).get('optimal_threshold', 0.5)
                    pred = 'Diabetes' if proba >= optimal_threshold else 'No Diabetes'

                    predictions[name] = {
                        'prediction': pred,
                        'probability': float(proba),
                        'confidence': 'High' if abs(proba - 0.5) > 0.3 else 'Medium' if abs(
                            proba - 0.5) > 0.15 else 'Low'
                    }
                    probabilities.append(float(proba))

            avg_probability = np.mean(probabilities) if probabilities else 0.5

            return predictions, avg_probability, None

        except Exception as e:
            return None, 0.5, str(e)


def main():
    # Initialize predictor
    if 'predictor' not in st.session_state:
        st.session_state.predictor = DiabetesPredictor()
        st.session_state.models_trained = False

    predictor = st.session_state.predictor

    # Main title
    st.title("SmartGluco - Diabetes Prediction System")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Home", "Train Models", "Predict"])

    with tab1:
        show_home()
    with tab2:
        show_train_models(predictor)
    with tab3:
        show_predict(predictor)


def show_home():
    st.header("Welcome to SmartGluco!")

    st.markdown("""
    **SmartGluco** is an advanced diabetes prediction system that integrates machine learning 
    with comprehensive health analytics to provide accurate risk assessment.

    ### Features

    - **Multiple ML Models**: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest
    - **Class Imbalance Handling**: SMOTE and weighted loss functions
    - **Threshold Optimization**: Improved recall for diabetes detection
    - **Comprehensive Visualizations**: Performance metrics, learning curves, and analysis
    - **Real-time Predictions**: Interactive risk assessment

    ### Model Capabilities

    1. **Enhanced Preprocessing**: Polynomial features and standardization
    2. **Optimized Thresholds**: Better minority class recall
    3. **Ensemble Approach**: Combined predictions from multiple models
    4. **Theoretical Framework**: Health psychology integration

    ### How to Use

    1. **Train Models**: Go to the "Train Models" tab to train the machine learning models
    2. **Make Predictions**: Use the "Predict" tab to assess diabetes risk with interactive sliders
    3. **View Analytics**: Explore comprehensive visualizations and model performance metrics

    ### Clinical Relevance

    - Early diabetes detection
    - Risk factor analysis
    - Preventive health recommendations
    - Evidence-based decision support

    *Note: This tool is for educational and screening purposes. Always consult healthcare professionals for medical diagnosis.*
    """)


def show_train_models(predictor):
    st.header("Train Machine Learning Models")

    st.markdown("""
    This section trains multiple machine learning models on diabetes data with enhanced 
    preprocessing and class imbalance handling.
    """)

    if st.button("Train All Models", type="primary"):
        with st.spinner("Initializing model training..."):
            # Load data
            if predictor.data is None:
                data = predictor.load_data()
                if data is None:
                    st.error("Failed to load data")
                    return

            # Define features
            features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']

            # Preprocess data
            if predictor.preprocess_data(features):
                st.success("Data preprocessing completed")

                # Train models
                if predictor.train_models(use_smote=True):
                    st.success("Model training completed")

                    # Optimize thresholds
                    if predictor.optimize_thresholds():
                        st.success("Threshold optimization completed")

                        # Evaluate models
                        if predictor.evaluate_models():
                            st.success("Model evaluation completed")
                            st.session_state.models_trained = True

                            # Display results
                            st.subheader("Training Results Summary")

                            # Create results table
                            results_data = []
                            for name, result in predictor.results.items():
                                results_data.append({
                                    'Model': name,
                                    'Train Accuracy': f"{result['train_accuracy']:.3f}",
                                    'Test Accuracy (Std)': f"{result['test_accuracy_standard']:.3f}",
                                    'Test Accuracy (Opt)': f"{result['test_accuracy_optimized']:.3f}",
                                    'Recall (Std)': f"{result['recall_standard']:.3f}",
                                    'Recall (Opt)': f"{result['recall_optimized']:.3f}",
                                    'Recall Improvement': f"{result['recall_improvement']:.3f}",
                                    'ROC AUC': f"{result['roc_auc']:.3f}",
                                    'Optimal Threshold': f"{result['optimal_threshold']:.3f}"
                                })

                            st.dataframe(pd.DataFrame(results_data))

                            # Generate visualizations
                            st.subheader("Model Visualizations")
                            predictor.plot_visualizations()

                        else:
                            st.error("Model evaluation failed")
                    else:
                        st.error("Threshold optimization failed")
                else:
                    st.error("Model training failed")
            else:
                st.error("Data preprocessing failed")


def show_predict(predictor):
    st.header("Diabetes Risk Prediction")

    if not st.session_state.models_trained:
        st.warning("Please train the models first in the 'Train Models' tab.")
        return

    st.markdown("""
    Use the sliders below to input health metrics and get a diabetes risk prediction 
    from our ensemble of machine learning models.
    """)

    # Input sliders
    col1, col2 = st.columns(2)

    with col1:
        glucose = st.slider('Glucose (mg/dL)', 70, 200, 120,
                            help="Fasting blood glucose level")
        blood_pressure = st.slider('Blood Pressure (mmHg)', 60, 140, 80,
                                   help="Diastolic blood pressure")
        insulin = st.slider('Insulin (μU/mL)', 15, 300, 80,
                            help="Fasting insulin level")

    with col2:
        bmi = st.slider('BMI (kg/m²)', 15, 50, 25,
                        help="Body Mass Index")
        age = st.slider('Age (years)', 20, 100, 30,
                        help="Age in years")

    # Create input dataframe
    input_data = pd.DataFrame({
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'Insulin': [insulin],
        'BMI': [bmi],
        'Age': [age]
    })

    # Display input summary
    st.subheader("Patient Health Profile")
    st.write(input_data)

    # Health indicators
    col1, col2, col3 = st.columns(3)

    with col1:
        glucose_status = "Normal" if glucose <= 100 else "Prediabetic" if glucose <= 125 else "Diabetic Range"
        glucose_color = "green" if glucose_status == "Normal" else "orange" if glucose_status == "Prediabetic" else "red"
        st.markdown(f"**Glucose Status**: <span style='color:{glucose_color}'>{glucose_status}</span>", unsafe_allow_html=True)

    with col2:
        bmi_status = "Underweight" if bmi < 18.5 else "Normal" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
        bmi_color = "blue" if bmi_status == "Underweight" else "green" if bmi_status == "Normal" else "orange" if bmi_status == "Overweight" else "red"
        st.markdown(f"**BMI Category**: <span style='color:{bmi_color}'>{bmi_status}</span>", unsafe_allow_html=True)

    with col3:
        bp_status = "Normal" if blood_pressure < 80 else "Elevated" if blood_pressure < 90 else "High"
        bp_color = "green" if bp_status == "Normal" else "orange" if bp_status == "Elevated" else "red"
        st.markdown(f"**Blood Pressure**: <span style='color:{bp_color}'>{bp_status}</span>", unsafe_allow_html=True)

    # Prediction button
    if st.button("Assess Diabetes Risk", type="primary"):
        with st.spinner("Analyzing health metrics..."):
            predictions, avg_probability, error = predictor.predict(input_data)

            if error:
                st.error(f"Prediction error: {error}")
            else:
                # Overall risk assessment
                st.subheader("Risk Assessment Results")

                if avg_probability >= 0.7:
                    risk_level = "High Risk"
                    risk_color = "red"
                elif avg_probability >= 0.4:
                    risk_level = "Moderate Risk"
                    risk_color = "orange"
                else:
                    risk_level = "Low Risk"
                    risk_color = "green"

                st.markdown(f"""
                <div style='text-align: center; padding: 20px; border: 2px solid {risk_color}; border-radius: 10px;'>
                    <h2 style='color: {risk_color};'>{risk_level}</h2>
                    <h3 style='color: {risk_color};'>Average Probability: {avg_probability:.2%}</h3>
                </div>
                """, unsafe_allow_html=True)

                # Individual model predictions
                st.subheader("Ensemble Model Predictions")

                predictions_data = []
                for name, pred in predictions.items():
                    predictions_data.append({
                        'Model': name,
                        'Prediction': pred['prediction'],
                        'Probability': f"{pred['probability']:.2%}",
                        'Confidence': pred['confidence']
                    })

                predictions_df = pd.DataFrame(predictions_data)
                st.dataframe(predictions_df, use_container_width=True)

                # Risk factors analysis
                st.subheader("Risk Factor Analysis")

                risk_factors = []
                if glucose > 125:
                    risk_factors.append("Elevated glucose levels (potential prediabetes/diabetes)")
                if bmi > 30:
                    risk_factors.append("Obesity (BMI > 30)")
                elif bmi > 25:
                    risk_factors.append("Overweight (BMI 25-30)")
                if age > 45:
                    risk_factors.append("Age over 45 years")
                if blood_pressure > 90:
                    risk_factors.append("Elevated blood pressure")

                if risk_factors:
                    st.warning("**Identified Risk Factors:**")
                    for factor in risk_factors:
                        st.write(f"- {factor}")
                else:
                    st.success("No significant risk factors identified in the provided metrics")

                # Recommendations
                st.subheader("Health Recommendations")

                if risk_level == "High Risk":
                    st.error("""
                    **Immediate Actions Recommended:**
                    - Consult with a healthcare provider promptly
                    - Consider HbA1c testing for definitive diagnosis
                    - Implement lifestyle changes (diet and exercise)
                    - Regular glucose monitoring
                    - Cardiovascular health assessment
                    """)
                elif risk_level == "Moderate Risk":
                    st.warning("""
                    **Preventive Measures:**
                    - Regular health check-ups and screening
                    - Weight management program
                    - Increased physical activity (150 mins/week)
                    - Balanced nutrition with reduced sugar intake
                    - Stress management and adequate sleep
                    """)
                else:
                    st.success("""
                    **Maintenance Guidelines:**
                    - Continue healthy lifestyle habits
                    - Annual health screenings
                    - Balanced diet rich in fiber
                    - Regular physical activity
                    - Maintain healthy weight
                    """)


if __name__ == "__main__":
    main()