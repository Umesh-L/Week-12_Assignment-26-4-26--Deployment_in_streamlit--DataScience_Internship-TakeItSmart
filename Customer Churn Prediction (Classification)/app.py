# create environment : python -m venv myenv
# activate environment : myenv\Scripts\activate
# install all libraries : pip install streamlit pandas numpy seaborn matplotlib scikit-learn
# to run the code : streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Load the single pipeline file ---
with open('customer_churn_pipeline.pkl', 'rb') as f:
    saved_data = pickle.load(f)

model = saved_data['model']
scaler = saved_data['scaler']
feature_cols = saved_data['features']

# --- APP UI ---
st.set_page_config(layout="wide")
st.title("Telecom Customer Churn Predictor")
st.write("Enter the customer's details below to predict their likelihood of canceling their service.")

# Create 3 columns for a clean UI layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics & Account")
    gender = st.selectbox('Gender', ('Female', 'Male'))
    SeniorCitizen = st.selectbox('Senior Citizen', (0, 1))
    Partner = st.selectbox('Partner', ('Yes', 'No'))
    Dependents = st.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.number_input('Tenure (Months)', min_value=0, max_value=100, value=12)

with col2:
    st.subheader("Services Subscribed")
    InternetService = st.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    PhoneService = st.selectbox('Phone Service', ('Yes', 'No'))
    MultipleLines = st.selectbox('Multiple Lines', ('No', 'Yes', 'No phone service'))
    TechSupport = st.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    OnlineSecurity = st.selectbox('Online Security', ('No', 'Yes', 'No internet service'))

with col3:
    st.subheader("Contract & Billing")
    Contract = st.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    PaymentMethod = st.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    PaperlessBilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
    MonthlyCharges = st.number_input('Monthly Charges ($)', min_value=10.0, max_value=200.0, value=70.0)
    TotalCharges = st.number_input('Total Charges ($)', min_value=0.0, max_value=10000.0, value=500.0)

st.write("---")

# --- PREDICTION LOGIC ---
if st.button('Predict Churn Risk', type="primary"):
    
    # 1. Initialize a dataframe of zeros with the exact columns used during training
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
    
    # 2. Fill Numerical Data
    input_df['SeniorCitizen'] = SeniorCitizen
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = MonthlyCharges
    input_df['TotalCharges'] = TotalCharges
    
    # 3. Handle background service states cleanly
    # If no internet, background services technically register as 'No internet service' in the dataset
    background_status = 'No internet service' if InternetService == 'No' else 'No'
    
    # 4. Fill Categorical Data (Dynamic One-Hot Encoding)
    categorical_mappings = {
        'gender': gender,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'TechSupport': TechSupport,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        # Secondary background services default to 'No' or 'No internet service' to keep UI simple
        'OnlineBackup': background_status,
        'DeviceProtection': background_status,
        'StreamingTV': background_status,
        'StreamingMovies': background_status
    }
    
    for feature, value in categorical_mappings.items():
        # Construct the column name exactly as pd.get_dummies did during training
        col_name = f"{feature}_{value}"
        # If the column exists (meaning it wasn't dropped by drop_first), set it to 1
        if col_name in input_df.columns:
            input_df[col_name] = 1
            
    # 5. Scale the input data
    scaled_input = scaler.transform(input_df)
    
    # 6. Predict using the Logistic Regression Model
    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1] * 100
    
    # 7. Display Result
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk of Churn!** There is a {probability:.1f}% chance this customer will cancel their service.")
    else:
        st.success(f"✅ **Low Risk of Churn.** There is only a {probability:.1f}% chance this customer will cancel.")