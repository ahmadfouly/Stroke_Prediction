import streamlit as st
import pandas as pd
from joblib import load
import pandas as pd

# Load your trained model from file
model = load("stroke_prediction_model.joblib")

st.title('Stroke Prediction App')
st.write("This app uses a machine learning model to predict the probability of a stroke based on the input data provided.")

# Define input fields
fields = {
    'gender': st.selectbox('Gender', ['Male', 'Female', 'Other']),
    'age': st.number_input('Age', min_value=0, max_value=100, value=50),
    'hypertension': st.selectbox('Hypertension', [0, 1]),
    'heart_disease': st.selectbox('Heart Disease', [0, 1]),
    'ever_married': st.selectbox('Ever Married', ['No', 'Yes']),
    'work_type': st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'Children']),
    'residence_type': st.selectbox('Residence Type', ['Rural', 'Urban']),
    'avg_glucose_level': st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0),
    'bmi': st.number_input('BMI', min_value=10.0, max_value=50.0, value=22.0),
    'smoking_status': st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'unknown'])
}

# Create DataFrame for prediction
input_df = pd.DataFrame([fields])

# Prediction button
if st.button('Predict Stroke'):
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0]
    st.write(f"Prediction: {'Stroke' if predictions[0] == 1 else 'No Stroke'}")
    st.write(f"Probability of not having a stroke: {probabilities[0]:.2f}")
    st.write(f"Probability of having a stroke: {probabilities[1]:.2f}")
    if predictions[0] == 1:
        st.error('Warning: High risk of stroke.')
    else:
        st.success('Low risk of stroke.')
