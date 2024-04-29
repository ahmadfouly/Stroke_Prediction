import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load your trained model and the fitted preprocessor from file
model = load_model('stroke_prediction_model.h5')
preprocessor = joblib.load('preprocessor.pkl')

st.title('Stroke Prediction App')
st.write("This app uses a machine learning model to predict the probability of a stroke based on the input data provided.")

# Define input fields (ensure these match exactly with those used during training)
input_data = {
    'gender': st.selectbox('Gender', ['Male', 'Female', 'Other']),
    'age': st.number_input('Age', min_value=0, max_value=100, value=50),
    'hypertension': st.selectbox('Hypertension', [0, 1]),
    'heart_disease': st.selectbox('Heart Disease', [0, 1]),
    'ever_married': st.selectbox('Ever Married', ['No', 'Yes']),
    'work_type': st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'Children']),
    'Residence_type': st.selectbox('Residence Type', ['Rural', 'Urban']),
    'avg_glucose_level': st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0),
    'bmi': st.number_input('BMI', min_value=10.0, max_value=50.0, value=22.0),
    'smoking_status': st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'unknown'])
}

# Button to make predictions
if st.button('Predict Stroke Risk'):
    # Create DataFrame from input fields
    input_df = pd.DataFrame([input_data])

    # Apply preprocessing
    transformed_input = preprocessor.transform(input_df)

    # Make prediction
    prediction = model.predict(transformed_input)
    probability = prediction[0][0]

    if probability > 0.2:  # Adjusted threshold for predicting high risk
        st.error('Warning: High risk of stroke.')
    else:
        st.success('Low risk of stroke.')
