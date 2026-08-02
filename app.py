import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("diabetes_model.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("🩺 Diabetes Prediction System")
st.write("Fill in the patient's details and click **Predict**.")

st.divider()

# -----------------------------
# Input Fields
# -----------------------------
pregnancies = st.number_input(
    "Number of Pregnancies",
    min_value=0,
    max_value=20,
    value=0
)

glucose = st.number_input(
    "Glucose Level",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure (mm Hg)",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    input_data = np.array([[pregnancies,
                            glucose,
                            blood_pressure,
                            skin_thickness,
                            insulin,
                            bmi,
                            diabetes_pedigree,
                            age]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ The model predicts that the person is **likely to have diabetes.**")
    else:
        st.success("✅ The model predicts that the person is **unlikely to have diabetes.**")
