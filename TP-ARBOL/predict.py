import cv2
import numpy as np
import joblib
import os

def calculate_hu_moments_from_frame(frame):
    """
    Calcula los momentos de Hu a partir de un frame de la cámara.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        moments = cv2.moments(cnt)
        if moments['m00'] != 0:
            hu_moments = cv2.HuMoments(moments)
            log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
            return log_hu_moments.flatten()
    return None

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    GENERATED_DIR = os.path.join(BASE_DIR, "generated-files")
    MODEL_PATH = os.path.join(GENERATED_DIR, "decision_tree_hu.joblib")
    ENCODER_PATH = os.path.join(GENERATED_DIR, "label_encoder.joblib")

    # Cargar el modelo y el codificador
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except FileNotFoundError:
        print("Error: No se encontraron los archivos del modelo. Asegúrate de haber ejecutado main.py primero.")
        return

    # Iniciar la cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir. Mostrando la cámara...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hu_features = calculate_hu_moments_from_frame(frame)

        if hu_features is not None:
            # Reajusta el array para que el modelo pueda hacer la predicción
            hu_features = hu_features.reshape(1, -1)
            
            prediction_encoded = model.predict(hu_features)
            prediction_text = encoder.inverse_transform(prediction_encoded)[0]
            
            # Muestra la predicción en la ventana de la cámara
            cv2.putText(frame, f"Prediccion: {prediction_text}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Muestra el frame
        cv2.imshow('Detector de Figuras', frame)

        # Si se presiona 'q', sale del bucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera la cámara y cierra las ventanas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()