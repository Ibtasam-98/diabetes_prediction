import pandas as pd


# In prediction.py, update the predict_new_data function:
def predict_new_data(model, scaler, poly):
    print("\n--- Enter Patient Information for Prediction ---")
    try:
        glucose = float(input("Enter Glucose (70-200 mg/dL): "))
        blood_pressure = float(input("Enter BloodPressure (60-140 mmHg): "))
        insulin = float(input("Enter Insulin (15-300 μU/mL): "))
        bmi = float(input("Enter BMI (15-50 kg/m²): "))
        age = float(input("Enter Age (20-100 years): "))

        # Validate input ranges
        if not (70 <= glucose <= 200):
            print("Warning: Glucose value outside typical range (70-200 mg/dL)")
        if not (60 <= blood_pressure <= 140):
            print("Warning: Blood pressure outside typical range (60-140 mmHg)")
        if not (15 <= insulin <= 300):
            print("Warning: Insulin outside typical range (15-300 μU/mL)")
        if not (15 <= bmi <= 50):
            print("Warning: BMI outside typical range (15-50 kg/m²)")
        if not (20 <= age <= 100):
            print("Warning: Age outside typical range (20-100 years)")

        user_data_df = pd.DataFrame([[glucose, blood_pressure, insulin, bmi, age]],
                                    columns=['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age'])
        user_data_scaled = scaler.transform(user_data_df)
        user_data_poly = poly.transform(user_data_scaled)
        prediction = model.predict(user_data_poly)

        if prediction[0] == 1:
            print("\nPrediction: The person is likely to have diabetes.")
        else:
            print("\nPrediction: The person is likely not to have diabetes.")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(user_data_poly)[0]
            print(f"Probability: No Diabetes: {proba[0]:.2%}, Diabetes: {proba[1]:.2%}")

    except ValueError as e:
        print(f"\nError: Invalid input - {e}. Please enter numeric values.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")