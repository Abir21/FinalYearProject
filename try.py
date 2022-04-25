import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import OneHotEncoder


header = st.container()
dataset = st.container()
features = st.container()
model = st.container()

#Creating title
with header:
    st.title('The FDP re-employement for UNHCR contribution Scheme')

#Creating feature inputs
with features:

    sellect_col, display_col = st.columns(2)
    lang = sellect_col.selectbox(
        'Please select language qualification:',
        ("make a selection", "Native", "Native + English", "Native + English + French", "Native + French")
    )
    
    
    
    edu_level = sellect_col.selectbox(
        'Please select latest Education Qualification:',
        ("make a selection", "No Education", "Primary", "High School", "Undergrad", "Associate Degree", "Vocational Degree", "Masters", "PhD")
    )
    
    
    last_occu = sellect_col.selectbox(
        'Please select Occupation before forced displacement:',
        ("make a selection", "Not employed","Chefs", "Baker", "Fisher", "Farmer", "Construction Worker", "Mechanic", "University Lecturer", "Software Developer", "Data Scientist", "AI Specialist", "DevOps Professional", 
                              "Machine Learning Professional", "Cybersecurity Professional", 
                              "Newtwork Engineer", "Cloud Engineer", "IT Professional and Constultant", 
                              "Dentist", "Vet", "Medical Doctor", "Medical Technician", 
                              "Physical Therapist", "Pharmaceutical Professional", 
                              "Nurses and Personal Carers", "Mechanical Engineer", "Electronic Engineer", "Civil Engineer", 
                              "Electrical Engineer", "Petroleum Engineer", "Accountant", "Sales Professional", 
                              "Legal Professional", "Marketing Professional", "Managment Professional", "Administrative Assistant")
    )
    
    #priting out the selections:
    st.write('Selected latest Education Qualification: ', edu_level)
    st.write('Selected language qualification: ', lang)
    st.write('Selected occupation before forced displacement: ', last_occu)
    
#Creating the data mapping for the onehotencoded columns:    
with dataset:
    #st.header('Synthetic skills and employement possibilities Dataset')
    #st.text('The Dataset have been made from various sources including UNHCR, Talent Beyond Boundaries Categories, Indeed, Glassdoor, Government Websites and many more')
    
    df = pd.read_csv('data/refugee_range_dataset.csv', index_col=0)
    #st.write(data.head())
    
    df1 = df[['language','education_level', 'last_occupation', 'mandatory_contribution_range']]
    df2 = df1.drop(['mandatory_contribution_range'], axis=1)
    
    #Mapping user inputs by using a dictionary:
    dict_input = {'language':lang, 'education_level':edu_level, 'last_occupation':last_occu}
    
    df_input = pd.DataFrame([dict_input])

    #OneHotEncoding column expansion:
    column_expansion_df = ['language_Native', 'language_Native + English',
       'language_Native + English + French', 'language_Native + French',
       'education_level_Associate Degree', 'education_level_High School',
       'education_level_Masters', 'education_level_No Education',
       'education_level_PhD', 'education_level_Primary',
       'education_level_Undergrad', 'education_level_Vocational Degree',
       'last_occupation_AI Specialist', 'last_occupation_Accountant',
       'last_occupation_Administrative Assistant', 'last_occupation_Baker',
       'last_occupation_Chefs', 'last_occupation_Civil Engineer',
       'last_occupation_Cloud Engineer', 'last_occupation_Construction Worker',
       'last_occupation_Cybersecurity Professional',
       'last_occupation_Data Scientist', 'last_occupation_Dentist',
       'last_occupation_DevOps Professional',
       'last_occupation_Electrical Engineer',
       'last_occupation_Electronic Engineer', 'last_occupation_Farmer',
       'last_occupation_Fisher',
       'last_occupation_IT Professional and Constultant',
       'last_occupation_Legal Professional',
       'last_occupation_Machine Learning Professional',
       'last_occupation_Managment Professional',
       'last_occupation_Marketing Professional', 'last_occupation_Mechanic',
       'last_occupation_Mechanical Engineer', 'last_occupation_Medical Doctor',
       'last_occupation_Medical Technician',
       'last_occupation_Newtwork Engineer', 'last_occupation_Not employed',
       'last_occupation_Nurses and Personal Carers',
       'last_occupation_Petroleum Engineer',
       'last_occupation_Pharmaceutical Professional',
       'last_occupation_Physical Therapist',
       'last_occupation_Sales Professional',
       'last_occupation_Software Developer',
       'last_occupation_University Lecturer', 'last_occupation_Vet']
    oneHot_df = pd.get_dummies(df_input).reindex(columns=column_expansion_df, fill_value=0)
    
    
    oneHot_df1 = pd.concat([oneHot_df], axis=1)
    
#Importing the trained model:
with model:
    #importing the model using joblib:
    contribute_model = joblib.load('model_folder/range_final_model.pkl')
    
    #creating pred function:
    def pred():
        model = contribute_model.predict(oneHot_df)
        return model
    
    contribute_p_btn = st.button('Predict Outcome', on_click=pred)
    
    if contribute_p_btn:
        outcome = pred()
        st.success(f'The skill values of the FDP is: {outcome}')