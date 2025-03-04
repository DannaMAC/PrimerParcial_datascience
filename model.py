import os
import tarfile
import urllib.request
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Descargar y extraer los datos
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("data")  
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz" 

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

# Cargar datos
csv_path = os.path.join(HOUSING_PATH, "housing.csv")
df = pd.read_csv(csv_path)

# Seleccionar características y variable objetivo
X = df[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income']]
y = df['median_house_value']

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo en la carpeta data
MODEL_PATH = os.path.join("data", "model.pkl")  # Guardar el modelo en la carpeta data
joblib.dump(model, MODEL_PATH)
print(f"✅ Modelo guardado en {MODEL_PATH}")
