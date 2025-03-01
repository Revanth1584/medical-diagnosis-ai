import streamlit as st
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")

# Streamlit UI
st.title("🩺 AI-Powered Diabetes Prediction")

st.write("Enter your health details to check for diabetes risk.")

# User Inputs
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

# Prediction
if st.button("Predict"):
    input_data = [[glucose, blood_pressure, skin_thickness]]
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ High risk of diabetes detected!")
    else:
        st.success("✅ No diabetes detected. Stay healthy!")
