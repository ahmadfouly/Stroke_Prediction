import streamlit as st
import numpy as np
from joblib import load

# Load your trained model and scaler from file
model = load("decision_tree_model.pkl")
scaler = load("scaler_dt.pkl")

st.title('Stroke Prediction App')  
st.write("This app uses a machine learning model to predict the probability of a stroke based on the input data provided.")

# Input fields for the user to fill out
age = st.number_input('Age', min_value=0, max_value=100, value=50)
avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0, max_value=300.0, value=100.0)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=22.0)

gender = st.selectbox('Gender', ['Male', 'Female'])
ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
work_type = st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'])
residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
smoking_status = st.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes'])
hypertension = st.selectbox('Hypertension', [1, 0])  # 1 for yes, 0 for no
heart_disease = st.selectbox('Heart Disease', [1, 0])  # 1 for yes, 0 for no

# One-hot encode the categorical data
input_features = np.zeros(20)
feature_indices = {
    'gender_Female': 0,
    'gender_Male': 1,
    'ever_married_No': 2,
    'ever_married_Yes': 3,
    'work_type_Govt_job': 4,
    'work_type_Never_worked': 5,
    'work_type_Private': 6,
    'work_type_Self-employed': 7,
    'work_type_children': 8,
    'Residence_type_Rural': 9,
    'Residence_type_Urban': 10,
    'smoking_status_formerly smoked': 11,
    'smoking_status_never smoked': 12,
    'smoking_status_smokes': 13,
    'hypertension_0': 14,
    'hypertension_1': 15,
    'heart_disease_0': 16,
    'heart_disease_1': 17,
    'avg_glucose_level': 18,
    'bmi': 19
}

# Mapping inputs to the input feature array
input_features[feature_indices['gender_Female']] = 1 if gender == 'Female' else 0
input_features[feature_indices['gender_Male']] = 1 if gender == 'Male' else 0
input_features[feature_indices['ever_married_No']] = 1 if ever_married == 'No' else 0
input_features[feature_indices['ever_married_Yes']] = 1 if ever_married == 'Yes' else 0
input_features[feature_indices['work_type_Govt_job']] = 1 if work_type == 'Govt_job' else 0
input_features[feature_indices['work_type_Never_worked']] = 1 if work_type == 'Never_worked' else 0
input_features[feature_indices['work_type_Private']] = 1 if work_type == 'Private' else 0
input_features[feature_indices['work_type_Self-employed']] = 1 if work_type == 'Self-employed' else 0
input_features[feature_indices['work_type_children']] = 1 if work_type == 'children' else 0
input_features[feature_indices['Residence_type_Rural']] = 1 if residence_type == 'Rural' else 0
input_features[feature_indices['Residence_type_Urban']] = 1 if residence_type == 'Urban' else 0
input_features[feature_indices['smoking_status_formerly smoked']] = 1 if smoking_status == 'formerly smoked' else 0
input_features[feature_indices['smoking_status_never smoked']] = 1 if smoking_status == 'never smoked' else 0
input_features[feature_indices['smoking_status_smokes']] = 1 if smoking_status == 'smokes' else 0
input_features[feature_indices['hypertension_0']] = 1 if hypertension == 0 else 0
input_features[feature_indices['hypertension_1']] = 1 if hypertension == 1 else 0
input_features[feature_indices['heart_disease_0']] = 1 if heart_disease == 0 else 0
input_features[feature_indices['heart_disease_1']] = 1 if heart_disease == 1 else 0
input_features[feature_indices['avg_glucose_level']] = avg_glucose_level
input_features[feature_indices['bmi']] = bmi

# Transform the input data using the scaler
input_features = scaler.transform([input_features])


# Prediction button
if st.button('Predict Stroke'):
    # Since it's a decision tree, you may choose to use just `.predict()` or `.predict_proba()`
    prediction = model.predict(input_features)
    if prediction[0] == 1:
        st.error('Warning: High risk of stroke.')
        probability_of_stroke = model.predict_proba(input_features)[0, 1]
        st.write(f"Probability of having a stroke: {probability_of_stroke:.2f}")
    else:
        st.success('Low risk of stroke.')
        probability_of_not_having_stroke = model.predict_proba(input_features)[0, 0]
        st.write(f"Probability of not having a stroke: {probability_of_not_having_stroke:.2f}")