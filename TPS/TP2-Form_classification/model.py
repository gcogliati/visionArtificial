# ...existing code...
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Ajustes para que el cambio del trackbar sea más notorio
BASE_THRESHOLD = 100
TRACKBAR_MAX = 30
TRACKBAR_SENSITIVITY = 8  # multiplicador para amplificar el efecto del trackbar

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
            y.append(label)
    return np.array(X), np.array(y)

def entrenar_modelo():
    """Entrena un modelo KNN usando el dataset."""
    X, y = cargar_dataset()
    modelo = KNeighborsClassifier(n_neighbors=3)
    modelo.fit(X, y)
    return modelo

# ---------------- RECONOCIMIENTO EN VIVO ----------------
def reconocimiento_en_vivo(modelo):
    """Abre la cámara y clasifica figuras en tiempo real con trackbar (0–30)."""
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

        # Obtener valor actual del trackbar (0–TRACKBAR_MAX) y centrarlo
        raw_val = get_trackbar_value("Threshold", window_name)
        centered = raw_val - mid  # puede ser negativo si se mueve hacia la izquierda

        # Amplificar el cambio para que sea más notorio
        scaled_offset = int(centered * TRACKBAR_SENSITIVITY)

        # Calcular threshold final y recortarlo al rango válido 0-255
        thresh_val = int(np.clip(BASE_THRESHOLD + scaled_offset, 0, 255))
        _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # filtro de ruido
                x, y_, w, h = cv2.boundingRect(cnt)
                roi = thresh[y_:y_+h, x:x+w]

                moments = cv2.moments(roi)
                hu = cv2.HuMoments(moments).flatten()
                for i in range(0, 7):
                    hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-10)

                prediccion = modelo.predict([hu])[0]

                cv2.rectangle(frame, (x, y_), (x+w, y_+h), (0,255,0), 2)
                cv2.putText(frame, prediccion, (x, y_-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (0,255,0), 2)

        cv2.imshow(window_name, frame)
        cv2.imshow("Threshold", thresh)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    modelo = entrenar_modelo()
    reconocimiento_en_vivo(modelo)
# ...existing code...