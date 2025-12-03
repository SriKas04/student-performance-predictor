import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.title("🎓 Student Performance Predictor")
st.write("Enter study profile to predict the final score.")

study_hours = st.number_input("Study Hours (per day)", min_value=0.0, max_value=16.0, value=3.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
past_score = st.number_input("Past Score (0–100)", min_value=0.0, max_value=100.0, value=65.0, step=1.0)

if st.button("Predict"):
    model = joblib.load("artifacts/student_performance_model.joblib")

    X = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "past_score": past_score
    }])

    pred = float(model.predict(X)[0])
    st.success(f"Predicted Final Score: {pred:.1f}")
