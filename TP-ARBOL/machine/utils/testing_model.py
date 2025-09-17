# testing_model.py

import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

def load_and_test(model_path, encoder_path, test_csv, fig_path):
    """
    Carga un modelo entrenado y un codificador, evalúa el modelo
    con un conjunto de datos de prueba y guarda una matriz de confusión.

    Args:
        model_path (str): Ruta al modelo entrenado (.joblib).
        encoder_path (str): Ruta al codificador de etiquetas (.joblib).
        test_csv (str): Ruta al archivo CSV con los datos de prueba.
        fig_path (str): Ruta donde se guardará la matriz de confusión.
    """
    print("Cargando modelo y codificador...")
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)

    # 1. Cargar los datos de prueba
    print("Cargando datos de prueba desde:", test_csv)
    df_test = pd.read_csv(test_csv)
    
    # 2. Separar características (X) y etiquetas (y)
    X_test = df_test.drop('class', axis=1)
    y_test_text = df_test['class']

    # 3. Codificar las etiquetas de prueba usando el mismo codificador
    y_test_encoded = encoder.transform(y_test_text)

    # 4. Hacer predicciones
    predictions = model.predict(X_test)
    
    # 5. Evaluar la precisión
    accuracy = accuracy_score(y_test_encoded, predictions)
    print(f"Precisión del modelo en el conjunto de prueba: {accuracy:.2f}")

    # 6. Generar y guardar la matriz de confusión
    cm = confusion_matrix(y_test_encoded, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión")
    plt.savefig(fig_path)
    print(f"Matriz de confusión guardada en: {fig_path}")