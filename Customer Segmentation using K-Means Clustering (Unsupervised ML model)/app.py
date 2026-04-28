# create environment : python -m venv myenv
# activate environment : myenv\Scripts\activate
# install all libraries : pip install streamlit pandas numpy seaborn matplotlib scikit-learn
# to run the code : streamlit run app.py

import streamlit as st
import pandas as pd
import pickle

# --- 1. Load the pipeline file ---
with open('customer_segmentation_pipeline.pkl', 'rb') as f:
    saved_data = pickle.load(f)

model = saved_data['model']
scaler = saved_data['scaler']
numeric_cols = saved_data['numeric_cols']
selected_features = saved_data['selected_features']

# --- APP UI ---
st.set_page_config(layout="wide")
st.title("Customer Segmentation Dashboard")
st.write("Enter a customer's purchasing behaviors to identify their marketing segment.")

# Create 2 columns for a clean UI layout
col1, col2 = st.columns(2)

with col1:
    annual_income = st.number_input('Annual Income ($)', min_value=10000, max_value=200000, value=50000, step=1000)
    purchase_amount = st.number_input('Average Purchase Amount ($)', min_value=0, max_value=2000, value=300, step=50)

with col2:
    purchase_frequency = st.number_input('Purchase Frequency (Visits/Year)', min_value=1, max_value=100, value=15)
    loyalty_score = st.slider('Loyalty Score (1.0 - 10.0)', min_value=1.0, max_value=10.0, value=5.0, step=0.1)

st.write("---")

# --- PREDICTION LOGIC ---
if st.button('Analyze Customer Segment', type="primary"):
    
    # 1. Create a dictionary with ALL 7 columns the scaler expects.
    # Use dummy values (1, 30, 0) for user_id, age, and region because the model ignores them anyway.
    input_dict = {
        'user_id': [1],
        'age': [30],
        'annual_income': [annual_income],
        'purchase_amount': [purchase_amount],
        'loyalty_score': [loyalty_score],
        'purchase_frequency': [purchase_frequency],
        'region_encoded': [0] 
    }
    
    # 2. Convert to dataframe ensuring column order matches notebook perfectly
    input_df = pd.DataFrame(input_dict)[numeric_cols]
    
    # 3. Scale the full 7-column dataframe
    scaled_array = scaler.transform(input_df)
    scaled_df = pd.DataFrame(scaled_array, columns=numeric_cols)
    
    # 4. Extract ONLY the 4 features the K-Means model needs
    final_input = scaled_df[selected_features]
    
    # 5. Predict the cluster (Segment 0, 1, 2, or 3)
    segment = model.predict(final_input)[0]
    
    # 6. Display Result
    st.success(f"🎯 **This customer belongs to Segment {segment}**")
    
   
    if segment == 0:
        st.success("💎 **Profile: VIP / Champion Customer**")
        st.write("These are your most valuable, highest-spending, and most loyal customers.")
        st.info("💡 **Marketing Action:** Do not offer discounts; they already buy. Offer exclusive perks, early access to new products, or invite them to a premium VIP loyalty tier.")
        
    elif segment == 1:
        st.success("📊 **Profile: Regular / Average Customer**")
        st.write("These customers have moderate income, average spend, and steady visit frequency.")
        st.info("💡 **Marketing Action:** Focus on upselling and cross-selling. Use targeted bundle offers to slightly increase their average cart value and push their loyalty score up.")
        
    elif segment == 2:
        st.warning("⚠️ **Profile: Casual / Low-Value Customer**")
        st.write("These customers have the lowest spend, lowest frequency, and lowest loyalty scores.")
        st.info("💡 **Marketing Action:** Highly price-sensitive. Send aggressive discount codes, flash sales, and re-engagement emails to incentivize them to visit more often.")
        
    elif segment == 3:
        st.success("⭐ **Profile: Rising Stars / Highly Loyal**")
        st.write("These customers spend frequently and have great loyalty, sitting just below the VIP tier.")
        st.info("💡 **Marketing Action:** Send personalized product recommendations based on past purchases and offer referral bonuses. Give them a small push to cross over into the VIP segment.")