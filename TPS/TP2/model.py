import cv2
import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ajustes para que el cambio del trackbar sea más notorio
BASE_THRESHOLD = 100
TRACKBAR_MAX = 30
TRACKBAR_SENSITIVITY = 8  # multiplicador para amplificar el efecto del trackbar
def label_to_int(string_label):
    if string_label == 'hearts': return 1
    if string_label == 'squares': return 2
    if string_label == 'stars':
        return 3

    else:
        raise Exception('unkown class_label')


def int_to_label(string_label):
    if string_label == 1: return 'hearts'
    if string_label == 2: return 'squares'
    if string_label == 3:
        return 'stars'
    else:
        raise Exception('unkown class_label')

# ---------------- TRACKBAR ----------------
def create_trackbar(trackbar_name, window_name, slider_max):
    # Inicializa el trackbar en el punto medio para permitir desplazamientos negativos/positivos
    cv2.createTrackbar(trackbar_name, window_name, slider_max // 2, slider_max, on_trackbar)

def on_trackbar(val):
    pass

def get_trackbar_value(trackbar_name, window_name):
    return int(cv2.getTrackbarPos(trackbar_name, window_name))

# ---------------- DATASET ----------------
def cargar_dataset(file="dataset.txt"):
    """Carga el dataset desde el archivo de texto."""
    X, y = [], []
    with open(file, "r") as f:
        for linea in f:
            parts = linea.strip().split(",")
            features = list(map(float, parts[:-1]))
            label = parts[-1]
            X.append(features)
            y.append(label_to_int(label))
    return np.array(X), np.array(y)

def entrenar_modelo():
    """Entrena un modelo KNN usando el dataset. Devuelve un Pipeline con scaler."""
    X, y = cargar_dataset()
    # Normalizar/estandarizar porque las componentes de Hu tienen escalas distintas
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=3))
    ])
    pipeline.fit(X, y)
    return pipeline

def save_model(model, filepath="model.pkl"):
    """Guarda el modelo entrenado en un archivo (joblib)."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Modelo guardado en {filepath}")

def load_model(filepath="model.pkl"):
    """Carga un modelo desde archivo. Lanza FileNotFoundError si no existe."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo de modelo: {filepath}")
    model = joblib.load(filepath)
    print(f"Modelo cargado desde {filepath}")
    return model

# ---------------- RECONOCIMIENTO EN VIVO ----------------
def reconocimiento_en_vivo(modelo):
    """Abre la cámara y clasifica figuras en tiempo real con trackbar (0–30).
    Extrae Hu moments desde la máscara rellenada del contorno (igual que en el dataset).
    """
    if modelo is None:
        raise Exception("Modelo no entrenado")

    cap = cv2.VideoCapture(0)
    window_name = "Reconocimiento de Figuras"
    cv2.namedWindow(window_name)

    # Crear trackbar para controlar el threshold (0–TRACKBAR_MAX)
    create_trackbar("Threshold", window_name, TRACKBAR_MAX)
    mid = TRACKBAR_MAX // 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)

        # Obtener valor actual del trackbar (0–TRACKBAR_MAX) y centrarlo
        raw_val = get_trackbar_value("Threshold", window_name)
        centered = raw_val - mid  # puede ser negativo si se mueve hacia la izquierda

        # Amplificar el cambio para que sea más notorio
        scaled_offset = int(centered * TRACKBAR_SENSITIVITY)

        # Calcular threshold final y recortarlo al rango válido 0-255
        thresh_val = int(np.clip(BASE_THRESHOLD + scaled_offset, 0, 255))
        _, thresh = cv2.threshold(blur, thresh_val, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # filtro de ruido
                x, y_, w, h = cv2.boundingRect(cnt)

                # Crear máscara del ROI con el contorno trasladado a coordenadas relativas
                mask_roi = np.zeros((h, w), dtype=np.uint8)
                # mover contorno al origen de la ROI
                cnt_shifted = cnt.copy()
                cnt_shifted = cnt_shifted - np.array([[x, y_]])
                cv2.drawContours(mask_roi, [cnt_shifted], -1, 255, thickness=-1)

                # Aplicar un closing ligero para homogeneizar (igual que en preprocessing)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
                mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, kernel, iterations=1)

                # Calcular Hu moments a partir de la máscara rellenada (coherente con el dataset)
                moments = cv2.moments(mask_roi)
                hu = cv2.HuMoments(moments).flatten()
                for i in range(0, 7):
                    hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-10)

                # Predecir usando el pipeline (incluye scaler)
                prediccion = modelo.predict([hu])[0]

                # Dibujar rectángulo, contorno y etiqueta en el frame (contorno en rojo)
                cv2.rectangle(frame, (x, y_), (x+w, y_+h), (0,255,0), 2)
                cv2.drawContours(frame, [cnt], -1, (0,0,255), 2)  # contorno detectado
                cv2.putText(frame, int_to_label(prediccion), (x, y_-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0,255,0), 2)

        cv2.imshow(window_name, frame)
        cv2.imshow("Threshold", thresh)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    modelo = entrenar_modelo()
    # guarda el modelo para exportarlo/usarlo luego
    save_model(modelo, "model.pkl")
    reconocimiento_en_vivo(modelo)