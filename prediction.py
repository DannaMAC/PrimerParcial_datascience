import pickle
import numpy as np
import pandas as pd

def load_model(model_path="model.pkl"):
    """Carga el modelo entrenado desde un archivo pickle."""
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

def preprocess_input(data):
    """Preprocesa los datos de entrada para que coincidan con el modelo."""
    df = pd.DataFrame([data])
    return df  # Ajusta según los pasos de preprocesamiento que usaste en el entrenamiento

def predict_price(model, input_data):
    """Realiza una predicción de precio a partir de los datos de entrada."""
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return prediction[0]

if __name__ == "__main__":
    # Cargar el modelo
    model = load_model()
    
    # Ejemplo de datos de entrada
    sample_input = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"  # Asegúrate de manejar variables categóricas correctamente
    }
    
    predicted_price = predict_price(model, sample_input)
    print(f"Precio estimado de la vivienda: ${predicted_price:.2f}")