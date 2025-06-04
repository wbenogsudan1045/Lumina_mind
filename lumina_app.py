import streamlit as st
import joblib

# Load model and encoders
model = joblib.load("lumina_model.pkl")
encoder = joblib.load("feature_encoder.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("🌟 LuminaMind - Coping Struggles Predictor")

# Inputs
growing_stress = st.selectbox("Growing Stress", ["Yes", "No"])
mood_swings = st.selectbox("Mood Swings", ["Yes", "No"])
mental_health_history = st.selectbox("Mental Health History", ["Yes", "No"])
changes_habits = st.selectbox("Changes in Habits", ["Yes", "No"])
work_interest = st.selectbox("Declining Interest in Work", ["Yes", "No"])
social_weakness = st.selectbox("Social Weakness", ["Yes", "No"])

if st.button("Predict Coping Status"):
    user_input = [[
        growing_stress,
        mood_swings,
        mental_health_history,
        changes_habits,
        work_interest,
        social_weakness
    ]]
    encoded_input = encoder.transform(user_input)
    prediction = model.predict(encoded_input)
    label = label_encoder.inverse_transform(prediction)[0]
    st.success(f"🧠 Predicted Coping Status: **{label}**")
