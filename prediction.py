import pandas as pd

def predict_new_data(model, scaler, poly):
    print("\n--- Enter Patient Information for Prediction ---")
    glucose = float(input("Enter Glucose: "))
    blood_pressure = float(input("Enter BloodPressure: "))
    insulin = float(input("Enter Insulin: "))
    bmi = float(input("Enter BMI: "))
    age = float(input("Enter Age: "))

    user_data_df = pd.DataFrame([[glucose, blood_pressure, insulin, bmi, age]],
                                columns=['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'Age'])
    user_data_scaled = scaler.transform(user_data_df)
    user_data_poly = poly.transform(user_data_scaled)
    prediction = model.predict(user_data_poly)

    if prediction[0] == 1:
        print("Prediction: The person is likely to have diabetes.")
    else:
        print("Prediction: The person is likely not to have diabetes.")