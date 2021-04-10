# import library
import streamlit as st
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
from sklearn.preprocessing import StandardScaler
import joblib


def load(dataset):
    data = pd.read_csv(dataset)
    return data

#create dictionary for categorical input
heart_disease_label = {'No':0,'Yes':1}
hypertensioin_label = {'No':0,'Yes':1}
smoking_status_label = {'smokes':0, 'formerly smoked':1, 
                        'Unknown':2, 'never smoked':3}
ever_maried_label = {'No':0,'Yes':1}
work_type_label = {'Never_worked':0,'Govt_job':1,'children':2,
                    'Self-employed':3,'Private':4}
gender_label = {'Other':0, 'Male':1, 'Female':2}
residence_type_label = {'Rural':0,'Urban':1}

#create get value function
def get_val(val,my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

# create function get keys
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key


image = Image.open('stroke_image.jfif')
Context = """Dataset Source: [Healthcare Dataset Stroke Data](https://www.kaggle.com/asaumya/healthcare-dataset-stroke-data) from Kaggle.\n
This dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, and various diseases and smoking status. A subset of the original train data is taken using the filtering method for Machine Learning and Data Visualization purposes.\n
About the Data: 
Each row in the data provides relavant information about a person , for instance; age, gender,smoking status, occurance of stroke in addition to other information
Unknown in Smoking status means the information is unavailable. N/A in other input fields imply that it is not applicable.\n
The Objective of this Project is to Create a model with 100 % F1 Score and AUC = 1, so the model is perfectly able to distinguish between positive class (have stroke) and negative class (not have stroke)."""

Link = 'https://github.com/tbreja/stroke-prediction-model.git'

Linkedin = 'https://www.linkedin.com/in/tbreja/'

# Main APP
def main():
    """Stroke Prediction Model"""  
    st.title('Stroke Predictor App')
    st.write('By : Tubagus Moh Reja')

    # Menu
    menu = ['Prediction', 'About this App']
    choiche = st.sidebar.selectbox('Select Activities', menu)

    if choiche == 'About this App':
        st.header('About this Project')
        st.image(image, width=570)
        st.write(Context)
        st.write(Link)
        st.write(Linkedin)
    if choiche == 'Prediction':
        st.image(image, width=600)
        # Making Widget input
        age = st.slider('How old are the patient ?', 0,100)
        heart_disease = st.radio('Does the patient has Heart Disease ?', tuple(heart_disease_label.keys()))
        hypertension = st.radio('Does the patient has Hypertension', tuple(hypertensioin_label.keys()))
        avg_glucose_level = st.slider('How much Average Glucose Level of the patient ?',0.00,300.00)
        smoking_status = st.radio("What is patient's smoking status ?", tuple(smoking_status_label.keys()))
        bmi = st.slider('How much Body Mass Index of the patient ?', 0, 100)
        ever_married = st.radio('Does the patient has ever married ?', tuple(ever_maried_label.keys()))
        gender = st.radio("What is the patient's gender ?", tuple(gender_label.keys()))
        work_type = st.radio("Work Type of the Patient ?", tuple(work_type_label.keys()))
        residence_type = st.radio("Residence Type where the patient's living ?", tuple(residence_type_label.keys()))

        # Encoding the input widget
        v_heart_disease = get_val(heart_disease, heart_disease_label)
        v_hypertension = get_val(hypertension,hypertensioin_label)
        v_smoking_status = get_val(smoking_status,smoking_status_label)
        v_ever_married = get_val(ever_married,ever_maried_label)
        v_gender = get_val(gender, gender_label)
        v_work_type = get_val(work_type,work_type_label)
        v_residence_type = get_val(residence_type,residence_type_label)
        v_avg_glucose_level = np.log(avg_glucose_level)


        # Compile the data for Forecasting
        input_data = [v_gender,age,v_hypertension,v_heart_disease,v_ever_married,v_work_type,v_residence_type,v_avg_glucose_level,bmi,v_smoking_status]
        input_data = np.array(input_data).reshape(1,-1)

        # Forecasting
        if st.button('Predict !'):
            predictor = pickle.load(open('stroke_predictor.pkl', 'rb'))
            predicting = predictor.predict(input_data)
            predict_result = (int(predicting))
            predict_proba = predictor.predict_proba(input_data)[:,0]
            proba_result = (str((np.around(float(predict_proba),3)*100)) + '%')
            def get(predict_result):
                result = str()
                if predict_result == 0:
                    result +="Doesn't Have Stroke"
                else:
                    result += 'Have Stroke'
                return result
            st.subheader('Prediction for The Patient is : ' + get(predict_result))
            st.subheader('The Probability for the Patient not have Stroke :' + proba_result)

    
if __name__ == '__main__':
    main()

