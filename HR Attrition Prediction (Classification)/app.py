# create environment : python -m venv myenv
# activate environment : myenv\Scripts\activate
# install all libraries : pip install streamlit pandas numpy seaborn matplotlib scikit-learn
# to run the code : streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Load the single pipeline file ---
with open('hr_attrition_pipeline.pkl', 'rb') as f:
    saved_data = pickle.load(f)

model = saved_data['model']
scaler = saved_data['scaler']
feature_cols = saved_data['features']

# --- APP UI ---
st.set_page_config(layout="wide")
st.title("HR Employee Attrition Predictor")
st.write("Enter the employee details below to predict their likelihood of leaving the company.")

# Create 3 columns for a clean UI layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    Age = st.number_input('Age', min_value=18, max_value=60, value=30)
    Gender = st.selectbox('Gender', ('Male', 'Female'))
    MaritalStatus = st.selectbox('Marital Status', ('Single', 'Married', 'Divorced'))
    Education = st.slider('Education Level (1-5)', 1, 5, 3)
    EducationField = st.selectbox('Education Field', ('Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Other', 'Human Resources'))
    DistanceFromHome = st.number_input('Distance From Home (miles)', 1, 50, 5)
    
with col2:
    st.subheader("Job Details")
    Department = st.selectbox('Department', ('Sales', 'Research & Development', 'Human Resources'))
    JobRole = st.selectbox('Job Role', ('Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director', 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'))
    JobLevel = st.slider('Job Level (1-5)', 1, 5, 2)
    BusinessTravel = st.selectbox('Business Travel', ('Travel_Rarely', 'Travel_Frequently', 'Non-Travel'))
    OverTime = st.selectbox('OverTime', ('Yes', 'No'))
    JobInvolvement = st.slider('Job Involvement (1-4)', 1, 4, 3)

with col3:
    st.subheader("Compensation & Tenure")
    MonthlyIncome = st.number_input('Monthly Income ($)', 1000, 25000, 5000)
    PercentSalaryHike = st.number_input('Percent Salary Hike (%)', 10, 30, 15)
    TotalWorkingYears = st.number_input('Total Working Years', 0, 40, 5)
    YearsAtCompany = st.number_input('Years At Company', 0, 40, 3)
    YearsInCurrentRole = st.number_input('Years In Current Role', 0, 20, 2)
    YearsWithCurrManager = st.number_input('Years With Curr Manager', 0, 15, 2)

st.write("---")

# --- PREDICTION LOGIC ---
if st.button('Predict Attrition Risk', type="primary"):
    
    # 1. Initialize a dataframe of zeros with the exact columns used during training
    # This cleverly handles drop_first=True missing columns automatically
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # 2. Fill Numerical Data (Using user inputs + standard background defaults)
    input_df['Age'] = Age
    input_df['DistanceFromHome'] = DistanceFromHome
    input_df['Education'] = Education
    input_df['JobInvolvement'] = JobInvolvement
    input_df['JobLevel'] = JobLevel
    input_df['MonthlyIncome'] = MonthlyIncome
    input_df['PercentSalaryHike'] = PercentSalaryHike
    input_df['TotalWorkingYears'] = TotalWorkingYears
    input_df['YearsAtCompany'] = YearsAtCompany
    input_df['YearsInCurrentRole'] = YearsInCurrentRole
    input_df['YearsWithCurrManager'] = YearsWithCurrManager
    
    # Standard Background Defaults
    input_df['DailyRate'] = 800
    input_df['HourlyRate'] = 65
    input_df['MonthlyRate'] = 14000
    input_df['NumCompaniesWorked'] = 2
    input_df['PerformanceRating'] = 3
    input_df['EnvironmentSatisfaction'] = 3
    input_df['JobSatisfaction'] = 3
    input_df['RelationshipSatisfaction'] = 3
    input_df['WorkLifeBalance'] = 3
    input_df['TrainingTimesLastYear'] = 2
    input_df['YearsSinceLastPromotion'] = 1
    input_df['StockOptionLevel'] = 0
    input_df['EmployeeCount'] = 1
    input_df['StandardHours'] = 80
    input_df['EmployeeNumber'] = 1000
    
    # 3. Fill Categorical Data (Dynamic One-Hot Encoding)
    categorical_mappings = {
        'Gender': Gender,
        'MaritalStatus': MaritalStatus,
        'EducationField': EducationField,
        'Department': Department,
        'JobRole': JobRole,
        'BusinessTravel': BusinessTravel,
        'OverTime': OverTime
    }
    
    for feature, value in categorical_mappings.items():
        # Construct the column name exactly as pd.get_dummies would have
        col_name = f"{feature}_{value}"
        # If the column exists (meaning it wasn't dropped by drop_first), set it to 1
        if col_name in input_df.columns:
            input_df[col_name] = 1
            
    # 4. Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # 5. Predict using the Logistic Regression Model
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1] * 100
    
    # 6. Display Result
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk of Attrition!** There is a {probability:.1f}% chance this employee will leave.")
    else:
        st.success(f"✅ **Low Risk of Attrition.** There is only a {probability:.1f}% chance this employee will leave.")