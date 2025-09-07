import cv2
import os

def preprocess_image(img_path, output_path=None):
    """Lee una imagen, elimina ruido, la binariza y opcionalmente la guarda."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Filtro Gaussiano para quitar ruido
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Umbralización (binarización)
    _, binary = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)

    # Operaciones morfológicas para limpiar puntos de ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if output_path:
        cv2.imwrite(output_path, clean)

    return clean

def preprocess_dataset(input_dir="figures", output_dir="figures_clean"):
    """Preprocesa todas las imágenes y guarda las limpias en otra carpeta."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for clase in os.listdir(input_dir):
        input_class = os.path.join(input_dir, clase)
        output_class = os.path.join(output_dir, clase)
        os.makedirs(output_class, exist_ok=True)

        for archivo in os.listdir(input_class):
            img_path = os.path.join(input_class, archivo)
            out_path = os.path.join(output_class, archivo)
            preprocess_image(img_path, out_path)

    print(f"Imágenes preprocesadas guardadas en {output_dir}")

if __name__ == "__main__":
    preprocess_dataset()