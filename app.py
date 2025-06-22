import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_excel("health care diabetes.xlsx")

# Prepare data
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit UI
st.title("🩺 Diabetic Patient Prediction")
st.write("Enter patient information to predict the likelihood of diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 120)
blood_pressure = st.number_input("Blood Pressure", 0, 140, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.number_input("Age", 0, 120, 30)

# Predict
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][prediction] * 100

    if prediction == 1:
        st.error(f"⚠️ The model predicts this person is likely diabetic. (Confidence: {probability:.2f}%)")
    else:
        st.success(f"✅ The model predicts this person is not diabetic. (Confidence: {probability:.2f}%)")
