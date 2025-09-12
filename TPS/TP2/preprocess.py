# ...existing code...
import cv2
import os
import numpy as np

def preprocess_image(img_path, output_path=None):
    """Lee una imagen, binariza y devuelve máscara preservando la forma real.
    Evita que el fondo o el padding se tomen como el objeto (que resultaba en
    un gran cuadrado blanco).
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Determinar color de fondo aproximado muestreando las esquinas
    corners = [gray[0,0], gray[0,-1], gray[-1,0], gray[-1,-1]]
    pad_color = int(np.median(corners))

    pad = 12
    gray_padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=int(pad_color))

    # Suavizado ligero
    blur = cv2.medianBlur(gray_padded, 3)

    # Otsu
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Asegurar que el objeto sea blanco (255) y fondo negro (0)
    white_area = int(np.sum(binary == 255))
    total = binary.size
    if white_area > (total // 2):
        binary = cv2.bitwise_not(binary)

    # Encontrar contornos en la imagen padded
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        clean = binary[pad:pad + gray.shape[0], pad:pad + gray.shape[1]]
        if output_path:
            cv2.imwrite(output_path, clean)
        return clean

    # Ordenar contornos por area descendente y escoger el primero "válido"
    padded_area = binary.shape[0] * binary.shape[1]
    min_valid_area = 200  # umbral para descartar ruido muy pequeño
    max_area_ratio = 0.98  # evitar contornos que ocupen casi toda la imagen

    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    chosen = None
    for cnt in contours_sorted:
        a = cv2.contourArea(cnt)
        if a < min_valid_area:
            continue
        if a > max_area_ratio * padded_area:
            # este contorno probablemente corresponde al fondo/padding o está pegado a bordes
            continue
        chosen = cnt
        break

    if chosen is None:
        # fallback: si no hay contorno "válido", tomar el segundo mayor si existe,
        # sino el mayor (pero sin rellenar toda la imagen)
        if len(contours_sorted) > 1:
            chosen = contours_sorted[1]
        else:
            chosen = contours_sorted[0]

    # Crear máscara y rellenar el contorno elegido
    mask = np.zeros_like(binary, dtype=np.uint8)
    cv2.drawContours(mask, [chosen], -1, 255, thickness=-1)

    # Aplicar un closing muy ligero para cerrar pequeños agujeros sin alterar puntas finas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Recortar padding
    clean = mask[pad:pad + gray.shape[0], pad:pad + gray.shape[1]]

    if output_path:
        cv2.imwrite(output_path, clean)

    return clean

def preprocess_dataset(input_dir="figures", output_dir="figures_clean"):
    """Preprocesa todas las imágenes y guarda las limpias en otra carpeta."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for clase in os.listdir(input_dir):
        input_class = os.path.join(input_dir, clase)
        if not os.path.isdir(input_class):
            continue
        output_class = os.path.join(output_dir, clase)
        os.makedirs(output_class, exist_ok=True)

        for archivo in os.listdir(input_class):
            img_path = os.path.join(input_class, archivo)
            out_path = os.path.join(output_class, archivo)
            preprocess_image(img_path, out_path)

    print(f"Imágenes preprocesadas guardadas en {output_dir}")

if __name__ == "__main__":
    preprocess_dataset()
# ...existing code...