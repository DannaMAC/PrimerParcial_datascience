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
    return df  

def predict_price(model, input_data):
    """Realiza una predicción de precio a partir de los datos de entrada."""
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return prediction[0]

if __name__ == "__main__":
    model = load_model()
    
    sample_input = {
        "longitude": 122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 8.0,
        "total_bedrooms": 12.0,
        "population": 32.0,
        "households": 120.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY"  
    }
    
    predicted_price = predict_price(model, sample_input)
    print(f"Precio estimado de la vivienda: ${predicted_price:.2f}")
