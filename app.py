import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

model_path = Path(__file__).parent / "car_price_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)


# Load your dataset (for options in dropdowns)
df = pd.read_csv(r"D:\ML Projects\Car-Price-Prediction-Using-LR\car data.csv")

df.drop_duplicates(inplace=True)
df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])
df['Car_Age'] = 2025 - df["Year"]

st.title("Car Selling Price Prediction 🚗💰")

st.sidebar.header("Input Car Details")

def user_input_features():
    Present_Price = st.sidebar.number_input("Present Price (in Lakhs)", min_value=0.0, step=0.5)
    Driven_kms = st.sidebar.number_input("Driven Kms", min_value=0, step=500)
    Owner = st.sidebar.selectbox("Number of Previous Owners", [0,1,2,3])
    Fuel_Type = st.sidebar.selectbox("Fuel Type", df['Fuel_Type'].unique())
    Selling_type = st.sidebar.selectbox("Selling Type", df['Selling_type'].unique())
    Transmission = st.sidebar.selectbox("Transmission", df['Transmission'].unique())
    Brand = st.sidebar.selectbox("Car Brand", df['Brand'].unique())
    Year = st.sidebar.number_input("Year of Manufacturing", min_value=1990, max_value=2025, step=1)

    Car_Age = 2025 - Year
    Price_per_Year = Present_Price / Car_Age
    Log_Driven_kms = np.log1p(Driven_kms)
    Price_Kms_Interaction = Present_Price * Driven_kms
    Is_Owner = 1 if Owner > 0 else 0
    
    data = {
        'Present_Price': Present_Price,
        'Driven_kms': Driven_kms,
        'Owner': Owner,
        'Fuel_Type': Fuel_Type,
        'Selling_type': Selling_type,
        'Transmission': Transmission,
        'Brand': Brand,
        'Car_Age': Car_Age,
        'Price_per_Year': Price_per_Year,
        'Log_Driven_kms': Log_Driven_kms,
        'Price_Kms_Interaction': Price_Kms_Interaction,
        'Is_Owner': Is_Owner
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)[0]

st.subheader("Predicted Selling Price")
st.write(f"💰 Estimated Selling Price: {prediction:.2f} Lakhs")

st.subheader("Car Input Details")
st.write(input_df)
st.write("Adjust the parameters in the sidebar to see how they affect the predicted selling price.")