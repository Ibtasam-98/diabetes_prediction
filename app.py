# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="SmartGluco - Diabetes Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load models and preprocessing objects
@st.cache_resource
def load_models():
    try:
        models = joblib.load('saved_models/models.pkl')
        scaler = joblib.load('saved_models/scaler.pkl')
        poly = joblib.load('saved_models/poly.pkl')
        threshold_results = joblib.load('saved_models/threshold_results.pkl')
        return models, scaler, poly, threshold_results
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None


models, scaler, poly, threshold_results = load_models()

# Define features
features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']

# App title and description
st.title("🩺 SmartGluco - Diabetes Risk Assessment")
st.markdown("""
### Integrating Machine Learning with Health Psychology Frameworks

This application uses advanced machine learning models enhanced with psychological theory
to provide comprehensive diabetes risk assessment based on health metrics.
""")

# Sidebar for user input
st.sidebar.header("Patient Information")


# Input fields with validation and theoretical framework context
def get_user_input():
    st.sidebar.markdown("### Health Metrics Input")

    glucose = st.sidebar.slider('Glucose (mg/dL)', 70, 200, 120,
                                help="Fasting blood glucose level - key indicator of diabetes risk")
    blood_pressure = st.sidebar.slider('Blood Pressure (mmHg)', 60, 140, 80,
                                       help="Diastolic blood pressure - cardiovascular health indicator")
    insulin = st.sidebar.slider('Insulin (μU/mL)', 15, 300, 80,
                                help="Fasting insulin level - measures insulin resistance")
    bmi = st.sidebar.slider('BMI (kg/m²)', 15, 50, 25,
                            help="Body Mass Index - obesity and metabolic syndrome indicator")
    age = st.sidebar.slider('Age (years)', 20, 100, 30,
                            help="Age - diabetes risk increases with age")

    # Create dictionary
    user_data = {
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'Insulin': insulin,
        'BMI': bmi,
        'Age': age
    }

    # Convert to DataFrame
    features_df = pd.DataFrame(user_data, index=[0])
    return features_df


user_input = get_user_input()

# Main panel
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Health Profile")
    st.write(user_input)

    # Health status indicators
    st.subheader("Health Status Indicators")

    # Glucose status
    glucose_status = "Normal" if user_input['Glucose'].iloc[0] <= 100 else "Prediabetic" if user_input['Glucose'].iloc[
                                                                                                0] <= 125 else "Diabetic Range"
    glucose_color = "green" if glucose_status == "Normal" else "orange" if glucose_status == "Prediabetic" else "red"

    # BMI status
    bmi_value = user_input['BMI'].iloc[0]
    if bmi_value < 18.5:
        bmi_status = "Underweight"
        bmi_color = "blue"
    elif bmi_value < 25:
        bmi_status = "Normal"
        bmi_color = "green"
    elif bmi_value < 30:
        bmi_status = "Overweight"
        bmi_color = "orange"
    else:
        bmi_status = "Obese"
        bmi_color = "red"

    st.markdown(f"**Glucose Level**: <span style='color:{glucose_color}'>{glucose_status}</span>",
                unsafe_allow_html=True)
    st.markdown(f"**BMI Category**: <span style='color:{bmi_color}'>{bmi_status}</span>", unsafe_allow_html=True)

with col2:
    st.markdown("### Theoretical Framework")
    st.markdown("""
    **Health Belief Model Integration:**
    - Personalized risk assessment
    - Enhanced perceived susceptibility
    - Motivational health feedback

    **Cognitive-Behavioral Approach:**
    - Real-time risk visualization
    - Behavioral reinforcement
    - Preventive health awareness
    """)

# Add some visual elements
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Clinical Reference Ranges")
    st.markdown("""
    - **Glucose**: 70-100 mg/dL (fasting, normal)
    - **Blood Pressure**: <120/80 mmHg (normal)
    - **Insulin**: 2-25 μU/mL (fasting, normal)
    - **BMI**: 18.5-24.9 kg/m² (healthy)
    - **Age**: Risk increases >45 years
    """)

with col2:
    st.markdown("### Diabetes Risk Factors")
    st.markdown("""
    • Elevated glucose levels
    • High blood pressure
    • Insulin resistance
    • Obesity (high BMI)
    • Advanced age
    • Family history
    • Sedentary lifestyle
    """)

# Enhanced Prediction section
st.sidebar.markdown("---")
st.sidebar.markdown("### Risk Assessment")

if st.sidebar.button('🔍 Assess Diabetes Risk', type='primary'):
    if models is None or scaler is None or poly is None:
        st.error("Models not loaded properly. Please check the model files.")
    else:
        try:
            # Preprocess input
            user_data_scaled = scaler.transform(user_input)
            user_data_poly = poly.transform(user_data_scaled)

            # Make predictions with all models using optimized thresholds
            predictions = {}
            risk_probabilities = []

            for model_name, model_data in models.items():
                # Get probability
                proba = model_data['model'].predict_proba(user_data_poly)[0][1]

                # Use optimized threshold if available
                optimal_threshold = model_data.get('optimal_threshold', 0.5)
                pred = 'Diabetes' if proba >= optimal_threshold else 'No Diabetes'

                predictions[model_name] = {
                    'prediction': pred,
                    'probability': float(proba),
                    'model_accuracy': float(
                        model_data.get('test_accuracy_optimized', model_data.get('test_accuracy_standard', 0.7))),
                    'recall': float(model_data.get('recall_optimized', 0.6)),
                    'optimal_threshold': float(optimal_threshold)
                }
                risk_probabilities.append(float(proba))

            # Enhanced consensus prediction with weighted voting
            avg_probability = np.mean(risk_probabilities)

            # Health Belief Model: Risk categorization with theoretical foundation
            if avg_probability >= 0.7:
                risk_level = "High Risk"
                risk_color = "red"
                theoretical_basis = "**Health Belief Model**: High perceived susceptibility detected. Strong recommendation for clinical consultation and lifestyle intervention."
            elif avg_probability >= 0.4:
                risk_level = "Moderate Risk"
                risk_color = "orange"
                theoretical_basis = "**Cognitive-Behavioral Approach**: Moderate risk suggests behavioral modifications could significantly reduce progression probability."
            else:
                risk_level = "Low Risk"
                risk_color = "green"
                theoretical_basis = "**Preventive Health Framework**: Current profile indicates low susceptibility. Maintain healthy habits for continued risk reduction."

            # Display enhanced results
            st.subheader("🎯 Comprehensive Risk Assessment")

            # Risk level with color coding
            st.markdown(f"### Overall Risk: <span style='color:{risk_color}; font-size:24px;'>{risk_level}</span>",
                        unsafe_allow_html=True)
            st.markdown(f"**Average Probability**: {avg_probability:.2%}")

            # Theoretical framework explanation
            st.markdown("### Psychological Framework Analysis")
            st.info(theoretical_basis)

            # Model predictions in an expanded table
            st.markdown("### Ensemble Model Predictions")
            predictions_df = pd.DataFrame.from_dict(predictions, orient='index')

            # Enhanced dataframe display
            styled_df = predictions_df.style.format({
                'probability': '{:.2%}',
                'model_accuracy': '{:.2%}',
                'recall': '{:.2%}',
                'optimal_threshold': '{:.3f}'
            }).background_gradient(subset=['probability'], cmap='RdYlGn_r')

            st.dataframe(styled_df, use_container_width=True)

            # Enhanced visualization
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Risk Probability Distribution")
                prob_df = pd.DataFrame({
                    'Model': predictions.keys(),
                    'Diabetes Probability': [p['probability'] for p in predictions.values()]
                })
                st.bar_chart(prob_df.set_index('Model'))

            with col2:
                st.markdown("### Clinical Recommendations")

                if risk_level == "High Risk":
                    st.error("""
                    **Immediate Actions Recommended:**
                    - Consult healthcare provider promptly
                    - Consider HbA1c testing
                    - Implement lifestyle changes
                    - Regular glucose monitoring
                    """)
                elif risk_level == "Moderate Risk":
                    st.warning("""
                    **Preventive Measures:**
                    - Regular health check-ups
                    - Weight management
                    - Physical activity
                    - Balanced nutrition
                    """)
                else:
                    st.success("""
                    **Maintenance Guidelines:**
                    - Continue healthy habits
                    - Annual health screenings
                    - Balanced diet
                    - Regular exercise
                    """)

            # Additional insights
            st.markdown("### Risk Factor Analysis")
            high_risk_factors = []
            if user_input['Glucose'].iloc[0] > 125:
                high_risk_factors.append("Elevated glucose levels")
            if user_input['BMI'].iloc[0] > 30:
                high_risk_factors.append("Obesity (High BMI)")
            if user_input['Age'].iloc[0] > 45:
                high_risk_factors.append("Advanced age")
            if user_input['BloodPressure'].iloc[0] > 90:
                high_risk_factors.append("Elevated blood pressure")

            if high_risk_factors:
                st.warning(f"**Notable risk factors**: {', '.join(high_risk_factors)}")
            else:
                st.info("No significant risk factors detected in provided metrics")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Enhanced footer with theoretical foundation
st.markdown("---")
st.markdown("### About SmartGluco")
st.markdown("""
**Theoretical Foundation Integration:**
- **Health Belief Model**: Enhances risk perception and preventive behavior motivation
- **Cognitive-Behavioral Framework**: Provides immediate feedback for behavior modification
- **Technology Acceptance Model**: Ensures user-friendly clinical implementation
- **Self-Determination Theory**: Supports autonomous health decision-making

**Machine Learning Models:**
- Logistic Regression (with class balancing)
- K-Nearest Neighbors
- Decision Tree (with enhanced recall)
- Support Vector Machine

**Advanced Features:**
- Class imbalance handling using SMOTE and weighted loss functions
- Threshold optimization for improved minority class recall
- Ensemble voting with theoretical framework integration
- Real-time risk assessment with psychological basis

*Trained on the Pima Indians Diabetes Dataset with enhanced preprocessing*
""")

# Enhanced disclaimer
st.markdown("---")
st.markdown("""
**Disclaimer**: This risk assessment tool integrates machine learning with health psychology frameworks
for educational and screening purposes. It should not replace professional medical diagnosis,
consultation, or treatment. Always consult with qualified healthcare providers for medical decisions.
""")
