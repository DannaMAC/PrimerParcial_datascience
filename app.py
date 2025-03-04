import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configurar la página con un fondo atractivo
st.set_page_config(page_title="Predicción de Precios de Casas", page_icon="🏡", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# CSS para mejorar el diseño
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            padding: 10px;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            border-radius: 10px;
            padding: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='text-align: center; color: #333;'>Predicción de Precios de Casas 🏡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555;'>Ingrese las características de la casa para predecir su precio.</p>", unsafe_allow_html=True)

# Diseño con columnas
col1, col2 = st.columns(2)

with col1:
    longitude = float(st.text_input("🌍 Longitud", "122.23"))
    latitude = float(st.text_input("🌎 Latitud", "37.88"))
    housing_median_age = float(st.text_input("🏠 Edad de la Vivienda", "41"))
    total_rooms = float(st.text_input("🛏️ Número Total de Habitaciones", "12"))

with col2:
    total_bedrooms = float(st.text_input("🛌 Número Total de Dormitorios", "12"))
    population = float(st.text_input("👥 Población", "32"))
    households = float(st.text_input("🏡 Número de Hogares", "120"))
    median_income = float(st.text_input("💰 Ingreso Mediano", "8.32"))

st.markdown("---")

# Botón con diseño
if st.button("🔍 Predecir Precio"):
    input_data = np.array([[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
                            population, households, median_income]])  
    prediction = model.predict(input_data)
    st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>💲 Precio estimado: ${prediction[0]:,.2f}</h2>", unsafe_allow_html=True)
