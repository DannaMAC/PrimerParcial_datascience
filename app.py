import streamlit as st
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Predicción de Precios de Casas 🏡")
st.write("Ingrese las características de la casa para predecir su precio.")

longitude = st.number_input("Longitud", value=122.23)
latitude = st.number_input("Latitud", value=37.88)
housing_median_age = st.number_input("Edad de la Vivienda", value=41)
total_rooms = st.number_input("Número Total de Habitaciones", value=12)
total_bedrooms = st.number_input("Número Total de Dormitorios", value=12)
population = st.number_input("Población", value=32)
households = st.number_input("Número de Hogares", value=120)
median_income = st.number_input("Ingreso Mediano", value=8.32)

if st.button("Predecir Precio"):
    input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
                            population, households, median_income]])  
    prediction = model.predict(input_data)
    st.success(f"El precio estimado de la casa es: ${prediction[0]:,.2f}")
