import joblib
import streamlit as st
import pandas as pd

diabetes_model = joblib.load('diabetes')

st.title('Diabetes Prediction using ML')

df = pd.read_csv('diabetes.csv')

col1, col2, col3 = st.columns(3)

with col1:
    Pregnancies = st.selectbox('Number of Pregnancies',sorted(df['Pregnancies'].unique()))
with col2:
       Glucose = st.selectbox('Glucose Level',sorted(df['Glucose'].unique()))
with col3:
    #BloodPressure = st.text_input('Blood Pressure value')
   # BloodPressure = st.selectbox('Blood Pressure value',df['BloodPressure'].unique())
    BloodPressure = st.selectbox('Blood Pressure value', sorted(df['BloodPressure'].unique()))

with col1:
    SkinThickness = st.selectbox('Skin Thickness value',sorted(df['SkinThickness'].unique()))

with col2:
    #Insulin = st.text_input('Insulin Level')
    Insulin = st.selectbox('Insulin Level',sorted(df['Insulin'].unique()))

with col3:
   # BMI = st.text_input('BMI value')
    BMI = st.selectbox('BMI value',sorted(df['BMI'].unique()))

with col1:
    #DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    DiabetesPedigreeFunction = st.selectbox('Diabetes Pedigree Function value',sorted(df['DiabetesPedigreeFunction'].unique()))

with col2:
    #Age = st.text_input('Age of the Person')
    Age = st.selectbox('Age of the Person',sorted(df['Age'].unique()))

diab_diagnosis = ''

# creating a button for Prediction
if st.button('Diabetes Test Result'):
    # Convert input values to numerical
    Pregnancies = int(Pregnancies)
    Glucose = int(Glucose)
    BloodPressure = int(BloodPressure)
    SkinThickness = int(SkinThickness)
    Insulin = int(Insulin)
    BMI = int(BMI)
    DiabetesPedigreeFunction = int(DiabetesPedigreeFunction)
    Age = int(Age)

    diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

    if diab_prediction[0] == 1:
        diab_diagnosis = 'The person is diabetic'
    else:
        diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)
