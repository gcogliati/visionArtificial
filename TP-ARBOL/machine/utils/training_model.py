# training_model.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def train_model(features_csv_path, model_path, encoder_path, cv_folds):
    """
    Entrena un modelo de clasificación de Árbol de Decisión usando datos
    de un CSV y guarda el modelo y el codificador de etiquetas.

    Args:
        features_csv_path (str): Ruta al archivo CSV con los datos de entrenamiento.
        model_path (str): Ruta donde se guardará el modelo entrenado.
        encoder_path (str): Ruta donde se guardará el codificador de etiquetas.
        cv_folds (int): Número de 'folds' para la validación cruzada.
    """
    # 1. Cargar los datos desde el archivo CSV
    df = pd.read_csv(features_csv_path)

    # 2. Separar características (X) y etiquetas (y)
    X = df.drop('class', axis=1)  # Las características son los 7 momentos de Hu
    y = df['class']  # La etiqueta es la clase de la figura (hearts, squares, etc.)

    # 3. Codificar las etiquetas de texto a números
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    # Guardar el codificador para usarlo en el test y en el futuro
    print(f"Codificador de etiquetas guardado en: {encoder_path}")
    joblib.dump(encoder, encoder_path)

    # 4. Entrenar el modelo de Árbol de Decisión
    print("Entrenando un modelo de Árbol de Decisión...")
    model = DecisionTreeClassifier()
    model.fit(X, y_encoded)

    # 5. Guardar el modelo entrenado
    print(f"Modelo de clasificación guardado en: {model_path}")
    joblib.dump(model, model_path)
    
    return model, encoder