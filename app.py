# app_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
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
        return models, scaler, poly
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None


models, scaler, poly = load_models()

# Define features
features = ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age']

# App title and description
st.title("Diabetes Prediction App")
st.markdown("""
This app predicts the likelihood of diabetes based on health metrics. 
Enter the patient's information in the sidebar and click 'Predict' to see the results.
""")

# Sidebar for user input
st.sidebar.header("Patient Information")


# Input fields with validation
def get_user_input():
    glucose = st.sidebar.slider('Glucose (mg/dL)', 70, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure (mmHg)', 60, 140, 80)
    insulin = st.sidebar.slider('Insulin (μU/mL)', 15, 300, 80)
    bmi = st.sidebar.slider('BMI (kg/m²)', 15, 50, 25)
    age = st.sidebar.slider('Age (years)', 20, 100, 30)

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
st.subheader("Patient Information")
st.write(user_input)

# Add some visual elements
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Normal Ranges")
    st.markdown("""
    - **Glucose**: 70-100 mg/dL (fasting)
    - **Blood Pressure**: <120/80 mmHg
    - **Insulin**: 2-25 μU/mL (fasting)
    - **BMI**: 18.5-24.9 kg/m²
    """)

with col2:
    st.markdown("### About Diabetes")
    st.markdown("""
    Diabetes is a chronic condition that affects how your body processes blood sugar (glucose).
    Early detection can help prevent complications.
    """)

# Prediction button
if st.sidebar.button('Predict'):
    if models is None or scaler is None or poly is None:
        st.error("Models not loaded properly. Please check the model files.")
    else:
        try:
            # Preprocess input
            user_data_scaled = scaler.transform(user_input)
            user_data_poly = poly.transform(user_data_scaled)

            # Make predictions with all models
            predictions = {}
            for model_name, model_data in models.items():
                pred = model_data['model'].predict(user_data_poly)[0]
                proba = model_data['model'].predict_proba(user_data_poly)[0][1]

                predictions[model_name] = {
                    'prediction': 'Diabetes' if pred == 1 else 'No Diabetes',
                    'probability': float(proba),
                    'model_accuracy': float(model_data['accuracy'])
                }

            # Determine consensus prediction
            diabetes_votes = sum(1 for pred in predictions.values() if pred['prediction'] == 'Diabetes')
            no_diabetes_votes = len(predictions) - diabetes_votes

            if diabetes_votes > no_diabetes_votes:
                consensus = 'Diabetes'
                consensus_color = 'red'
            elif no_diabetes_votes > diabetes_votes:
                consensus = 'No Diabetes'
                consensus_color = 'green'
            else:
                # If tie, use model with higher accuracy
                best_model = max(predictions.items(), key=lambda x: x[1]['model_accuracy'])
                consensus = best_model[1]['prediction']
                consensus_color = 'orange' if consensus == 'Diabetes' else 'green'

            # Display results
            st.subheader("Prediction Results")

            # Consensus prediction with color
            st.markdown(
                f"### Consensus Prediction: <span style='color:{consensus_color};font-weight:bold;'>{consensus}</span>",
                unsafe_allow_html=True)

            # Model predictions in a table
            st.markdown("### Individual Model Predictions")
            predictions_df = pd.DataFrame.from_dict(predictions, orient='index')
            st.dataframe(predictions_df.style.format({
                'probability': '{:.2%}',
                'model_accuracy': '{:.2%}'
            }))

            # Visual indicators
            st.markdown("### Probability Visualization")

            # Create a bar chart of probabilities
            prob_df = pd.DataFrame({
                'Model': predictions.keys(),
                'Diabetes Probability': [p['probability'] for p in predictions.values()]
            })

            st.bar_chart(prob_df.set_index('Model'))

            # Add some explanatory text
            if consensus == 'Diabetes':
                st.warning(
                    "The models suggest a high likelihood of diabetes. Please consult with a healthcare professional for further evaluation.")
            else:
                st.success(
                    "The models suggest a low likelihood of diabetes. However, regular check-ups are recommended for maintaining good health.")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Add some additional information
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This application uses machine learning models to predict diabetes risk based on:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Support Vector Machine (SVM)**

The models were trained on the Pima Indians Diabetes Dataset.
""")

# Add a footer
st.markdown("---")
st.markdown("""
*Note: This prediction is for informational purposes only and should not replace professional medical advice.*
""")