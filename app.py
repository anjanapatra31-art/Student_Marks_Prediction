import streamlit as st
import joblib
import pandas as pd

# Load trained model
model = joblib.load("student_marks_model.pkl")

# Website heading
st.title("Student Marks Prediction")

st.write("Predict student marks using Machine Learning")

# Student input
hours = st.number_input(
    "Hours Studied",
    min_value=0.0,
    max_value=24.0,
    value=5.0
)

previous_marks = st.number_input(
    "Previous Marks",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

sleep = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

# Prediction
if st.button("Predict Marks"):

    input_data = pd.DataFrame(
        [[hours, previous_marks, sleep]],
        columns=[
            "Hours_Studied",
            "Previous_Marks",
            "Sleep_Hours"
        ]
    )

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Marks: {prediction[0]:.2f}"
    )