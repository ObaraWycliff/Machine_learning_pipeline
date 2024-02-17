import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib


# Load the trained model
model = joblib.load("best_model_one.pkl")
# Define the app title and layout
st.title("VERSICOLOR FLOWER SPECIES PREDICTOR ")
st.write ("This application analyses  whether flower features  entered are of species veriscolor or not")

# Define input fields for features
sepal_length = st.number_input("sepal_length ", min_value=0.00, max_value=20.00, value=10.0, step=0.1)
sepal_width = st.number_input("sepal_width", min_value=0.00, max_value=20.00, value=10.0, step=0.1)
petal_length = st.number_input("petal_length", min_value=0.00, max_value=20.00, value=10.0, step=0.1)
petal_width = st.number_input("petal_width", min_value=0.00, max_value=20.00, value=10.0, step=0.1)


# Create a button for making predictions
if st.button("Predict"):
    # Process input values
    input_data = pd.DataFrame(
        {
            "sepal_length": [sepal_length],
            "sepal_width": [sepal_width],
            "petal_length": [petal_length],
            "petal_width": [petal_width],
        }
    )

    # Scale input data using the same scaler used during training
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    # Make a prediction using the trained model
    prediction = model.predict(input_data_scaled)

    # Display the prediction
    if prediction == 1:
        st.success("Species versicolor.")
    else:
        st.success("Species not versicolor.")
