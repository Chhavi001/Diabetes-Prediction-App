
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page config
st.set_page_config(page_title="🩺 Diabetes Prediction", layout="wide")
st.title("🩺 Diabetes Prediction App")
st.markdown("Check your risk of diabetes using this AI-based prediction tool.")

# Sidebar inputs
st.sidebar.header("Patient Details")
preg = st.sidebar.number_input("Pregnancies", 0, 20)
glucose = st.sidebar.number_input("Glucose Level", 0, 300)
bp = st.sidebar.number_input("Blood Pressure", 0, 200)
skin = st.sidebar.number_input("Skin Thickness", 0, 100)
insulin = st.sidebar.number_input("Insulin Level", 0, 900)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.sidebar.number_input("Age", 1, 120)

# Prediction button
if st.sidebar.button("🔍 Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(input_data)
    if result[0] == 1:
        st.error("😷 You may be Diabetic!")
    else:
        st.success("✅ You are Not Diabetic!")
else:
    st.info("Enter the details on the sidebar and click Predict.")
