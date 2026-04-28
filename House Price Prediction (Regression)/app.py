# create environment : python -m venv myenv
# activate environment : myenv\Scripts\activate
# install all libraries : pip install streamlit pandas numpy seaborn matplotlib scikit-learn
# to run the code : streamlit run app.py

import pickle
import pandas as pd
import numpy as np
import streamlit as st

# 1. Load the pipeline pickle file
with open('house_price_pipeline.pkl', 'rb') as f:
    saved_data = pickle.load(f)

# 2. Extract the model and scaler from the loaded dictionary
model = saved_data['model']
scaler = saved_data['scaler']

# --- APP UI ---
st.title("Real Estate House Price Predictor")
st.write("Enter the property details below to estimate its market price.")

# Input fields 
square_footage = st.number_input('Square Footage', min_value=500, max_value=15000, value=2000)
num_bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
num_bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
year_built = st.number_input('Year Built', min_value=1800, max_value=2024, value=2000)
lot_size = st.number_input('Lot Size (Acres)', min_value=0.0, max_value=20.0, value=0.5)
garage_size = st.number_input('Garage Size (Car Capacity)', min_value=0, max_value=5, value=2)
neighborhood_quality = st.slider('Neighborhood Quality (1-10)', min_value=1, max_value=10, value=5)

# --- PREDICTION LOGIC ---
if st.button('Predict House Price'):
    # Create input dataframe ensuring column names perfectly match your training data
    input_data = pd.DataFrame({
        'Square_Footage': [square_footage],
        'Num_Bedrooms': [num_bedrooms],
        'Num_Bathrooms': [num_bathrooms],
        'Year_Built': [year_built],
        'Lot_Size': [lot_size],
        'Garage_Size': [garage_size],
        'Neighborhood_Quality': [neighborhood_quality]
    })
    
    # Scale the inputs using the saved scaler
    scaled_input = scaler.transform(input_data)
    
    # Predict the log price using the saved model
    log_prediction = model.predict(scaled_input)
    
    # Convert log price back to actual dollars (anti-log)
    actual_price = np.expm1(log_prediction[0])
    
    # Display Result
    st.success(f"The estimated price for this house is: ${actual_price:,.2f}")