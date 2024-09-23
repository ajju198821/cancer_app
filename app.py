import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# load the model
model = tf.keras.models.load_model('123model.keras')

def pridict_cancer(input_data):
    input_df=pd.DataFrame(input_data, index=[0])
    scaler = MinMaxScaler()
    input_array = scaler.fit_transform(input_df)  # Directly use the transformed array
    predictions = model.predict(input_array)
    predicted_classes = (predictions > 0.5).astype("int32")  # Assuming a threshold of 0.5
    class_mapping = {0: 'Benign', 1: 'Malignant'}
    return class_mapping[predicted_classes[0][0]]

st.title("Breast Cancer Wisconsin (Diagnostic) using ANN Algorithm by TensorFlow")

# Create input fields for all 30 features
radius_mean = st.number_input("Radius Mean", value=14.0)
texture_mean = st.number_input("Texture Mean", value=20.0)
perimeter_mean = st.number_input("Perimeter Mean", value=90.0)
area_mean = st.number_input("Area Mean", value=600.0)
smoothness_mean = st.number_input("Smoothness Mean", value=0.1)
compactness_mean = st.number_input("Compactness Mean", value=0.15)
concavity_mean = st.number_input("Concavity Mean", value=0.2)
concave_points_mean = st.number_input("Concave Points Mean", value=0.1)
symmetry_mean = st.number_input("Symmetry Mean", value=0.2)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", value=0.06)
radius_se = st.number_input("Radius SE", value=0.2)
texture_se = st.number_input("Texture SE", value=1.0)
perimeter_se = st.number_input("Perimeter SE", value=1.5)
area_se = st.number_input("Area SE", value=20.0)
smoothness_se = st.number_input("Smoothness SE", value=0.005)
compactness_se = st.number_input("Compactness SE", value=0.02)
concavity_se = st.number_input("Concavity SE", value=0.03)
concave_points_se = st.number_input("Concave Points SE", value=0.01)
symmetry_se = st.number_input("Symmetry SE", value=0.03)
fractal_dimension_se = st.number_input("Fractal Dimension SE", value=0.004)
radius_worst = st.number_input("Radius Worst", value=16.0)
texture_worst = st.number_input("Texture Worst", value=25.0)
perimeter_worst = st.number_input("Perimeter Worst", value=105.0)
area_worst = st.number_input("Area Worst", value=800.0)
smoothness_worst = st.number_input("Smoothness Worst", value=0.12)
compactness_worst = st.number_input("Compactness Worst", value=0.2)
concavity_worst = st.number_input("Concavity Worst", value=0.3)
concave_points_worst = st.number_input("Concave Points Worst", value=0.15)
symmetry_worst = st.number_input("Symmetry Worst", value=0.25)
fractal_dimension_worst = st.number_input("Fractal Dimension Worst", value=0.08)


# Create a dictionary of input data
input_data = {
    'radius_mean': radius_mean,
    'texture_mean': texture_mean,
    'perimeter_mean': perimeter_mean,
    'area_mean': area_mean,
    'smoothness_mean': smoothness_mean,
    'compactness_mean': compactness_mean,
    'concavity_mean': concavity_mean,
    'concave_points_mean': concave_points_mean,
    'symmetry_mean': symmetry_mean,
    'fractal_dimension_mean': fractal_dimension_mean,
    'radius_se': radius_se,
    'texture_se': texture_se,
    'perimeter_se': perimeter_se,
    'area_se': area_se,
    'smoothness_se': smoothness_se,
    'compactness_se': compactness_se,
    'concavity_se': concavity_se,
    'concave_points_se': concave_points_se,
    'symmetry_se': symmetry_se,
    'fractal_dimension_se': fractal_dimension_se,
    'radius_worst': radius_worst,
    'texture_worst': texture_worst,
    'perimeter_worst': perimeter_worst,
    'area_worst': area_worst,
    'smoothness_worst': smoothness_worst,
    'compactness_worst': compactness_worst,
    'concavity_worst': concavity_worst,
    'concave_points_worst': concave_points_worst,
    'symmetry_worst': symmetry_worst,
    'fractal_dimension_worst': fractal_dimension_worst
}


# Create a button to predict the class
if st.button("Predict"):
    prediction = pridict_cancer(input_data)
    st.write("Prediction:", {prediction})