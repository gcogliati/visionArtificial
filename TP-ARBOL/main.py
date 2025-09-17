import cv2
import numpy as np
import os

# Rutas a las imágenes de referencia
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(BASE_DIR, "figures", "raw", "train")
CLASSES = ['hearts', 'squares', 'stars']

# Variables globales para las barras deslizantes
# Los valores iniciales son solo para la creación de las barras
threshold_value = 150
structuring_element_size = 5
max_distance = 0.5

# Diccionario para almacenar los contornos de referencia
reference_contours = {}

def load_reference_contours():
    """Carga los contornos de las imágenes de referencia."""
    for class_name in CLASSES:
        class_path = os.path.join(REF_DIR, class_name)
        image_files = [f for f in os.listdir(class_path) if f.endswith(('png', 'jpg', 'jpeg'))]
        if image_files:
            image_path = os.path.join(class_path, image_files[0])
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    reference_contours[class_name] = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    print("Contornos de referencia cargados:", reference_contours.keys())

def on_trackbar(value):
    """Función de callback para las barras deslizantes. No hace nada, ya que los valores se leen en el bucle principal."""
    pass

def create_trackbars():
    """Crea las barras deslizantes para ajustar los parámetros."""
    cv2.namedWindow('Detector de Formas')
    cv2.createTrackbar('Threshold', 'Detector de Formas', threshold_value, 255, on_trackbar)
    cv2.createTrackbar('SE Size', 'Detector de Formas', structuring_element_size, 20, on_trackbar)
    cv2.createTrackbar('Max Dist', 'Detector de Formas', int(max_distance * 100), 100, on_trackbar)

def process_frame(frame, threshold_val, se_size_val, max_dist_val):
    """Procesa un solo frame para detectar y clasificar figuras."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Aplicar umbralización
    _, thresh = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY_INV)

    # Operaciones morfológicas para eliminar ruido
    kernel = np.ones((se_size_val, se_size_val), np.uint8)
    processed_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Encontrar todos los contornos
    contours, _ = cv2.findContours(processed_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_frame = frame.copy()
    
    for contour in contours:
        # Filtrar contornos pequeños
        if cv2.contourArea(contour) < 100:
            continue
            
        # Comparar contorno con las figuras de referencia
        best_match = None
        min_distance = float('inf')
        
        for class_name, ref_contour in reference_contours.items():
            distance = cv2.matchShapes(contour, ref_contour, cv2.CONTOURS_MATCH_I1, 0.0)
            
            if distance < min_distance and distance < max_dist_val:
                min_distance = distance
                best_match = class_name
        
        # Anotar la imagen
        if best_match:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                cv2.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)
                cv2.putText(result_frame, best_match, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result_frame

def main():
    load_reference_contours()
    create_trackbars()
    
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    print("Presiona 'q' para salir.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lee los valores de las barras en cada iteración del bucle
        threshold_val = cv2.getTrackbarPos('Threshold', 'Detector de Formas')
        se_size_val = cv2.getTrackbarPos('SE Size', 'Detector de Formas')
        max_dist_val = cv2.getTrackbarPos('Max Dist', 'Detector de Formas') / 100.0

        # Asegura que el tamaño del elemento estructural sea al menos 1
        se_size_val = max(1, se_size_val)

        annotated_frame = process_frame(frame, threshold_val, se_size_val, max_dist_val)
        
        cv2.imshow('Detector de Formas', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()