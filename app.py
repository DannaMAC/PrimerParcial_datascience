import streamlit as st
import joblib
import numpy as np
from prediction import predict_price  # Importar la función de predicción

# Cargar el modelo entrenado
model = joblib.load("data/final_model.pkl")

st.title("Predicción de Precios de Casas")

# Definir las entradas
st.sidebar.header("Ingresa los datos de la casa")

longitude = st.sidebar.number_input("Longitud", value=-118.25)
latitude = st.sidebar.number_input("Latitud", value=34.05)
housing_median_age = st.sidebar.number_input("Edad Media de la Casa", value=30)
total_rooms = st.sidebar.number_input("Número Total de Habitaciones", value=2000)
total_bedrooms = st.sidebar.number_input("Número Total de Dormitorios", value=500)
population = st.sidebar.number_input("Población en la Zona", value=800)
households = st.sidebar.number_input("Número de Hogares", value=400)
median_income = st.sidebar.number_input("Ingreso Medio", value=3.5)

# Crear un array con los datos ingresados
input_data = [longitude, latitude, housing_median_age, total_rooms,
              total_bedrooms, population, households, median_income]

# Hacer la predicción cuando el usuario presiona el botón
if st.sidebar.button("Predecir Precio"):
    prediction = predict_price(model, input_data)
    st.write(f"**Precio estimado de la casa:** ${prediction:,.2f}")