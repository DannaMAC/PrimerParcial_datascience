# Predicción de Precios de Viviendas

Este proyecto es una aplicación interactiva desarrollada con Streamlit que utiliza un modelo de Machine Learning para predecir el precio de una vivienda basado en diversas características.

# Autor
Danna Corral 358147

## Instalación

1. Clona este repositorio:
   ```sh
   git clone https://github.com/DannaMAC/DATA-SCIENCE.git
   cd DATA-SCIENCE
   ```
2. Instala las dependencias:
   ```sh
   pip install -r requirements.txt
   ```
3. Descarga los datos:
   ```sh
   python model.py
   ```
4. Ejecuta la aplicación:
   ```sh
   streamlit run app.py
   ```

## Archivos Principales

- `app.py`: Interfaz gráfica con Streamlit.
- `model.py`: Script para entrenar y guardar el modelo.
- `prediction.py`: Carga el modelo entrenado y realiza predicciones.
- `requirements.txt`: Lista de librerías necesarias.
- `model.pkl`: Modelo de Machine Learning guardado.

## Modelo de Machine Learning
El modelo es un regresor entrenado con datos de viviendas, donde se consideran variables como:
- Ubicación (longitud y latitud)
- Edad de la vivienda
- Número de habitaciones
- Población en la zona
- Ingreso medio

## Uso de la Aplicación

1. Inicia la aplicación con `streamlit run app.py`.
2. Introduce los valores de la vivienda en la interfaz.
3. Haz clic en "Predecir" para obtener una estimación del precio.

## Tecnologías Usadas
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy

## Licencia
Este proyecto está bajo la licencia MIT. 