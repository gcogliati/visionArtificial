# ...existing code...
import cv2
import numpy as np
import os

def extraer_hu_moments(img):
    """Calcula los momentos de Hu de una imagen binaria."""
    moments = cv2.moments(img)
    huMoments = cv2.HuMoments(moments).flatten()
    # Escalado logarítmico
    for i in range(0, 7):
        huMoments[i] = -1 * np.sign(huMoments[i]) * np.log10(abs(huMoments[i]) + 1e-10)
    return huMoments

# ...existing code...
def extraer_hu_y_contorno(img):
    """
    Devuelve (huMoments, contour_mask, contour_points, vis_image).
    - img: imagen en escala de grises (puede ser binaria o no).
    - Se umbraliza automáticamente usando Otsu si la imagen no es puramente 0/255.
    - Selecciona el contorno de mayor área.
    - vis_image es la imagen en color con el contorno dibujado (BGR).
    """
    # Asegurar binario: si ya es sólo 0/255, lo usamos; si no, Otsu
    unique_vals = np.unique(img)
    if set(unique_vals.tolist()).issubset({0, 255}):
        thresh = img.copy()
    else:
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Encontrar contornos (trabajar sobre una copia)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None, None

    # Seleccionar el contorno más grande
    cnt = max(contours, key=cv2.contourArea)

    # Crear máscara del contorno (rellena)
    mask = np.zeros_like(thresh)
    cv2.drawContours(mask, [cnt], -1, 255, thickness=-1)

    # Calcular momentos a partir de la máscara (más robusto)
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()
    for i in range(7):
        hu[i] = -1 * np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-10)

    # Imagen de visualización: convertir a BGR y dibujar contorno en rojo
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, [cnt], -1, (0, 0, 255), 2)

    return hu, mask, cnt, vis

def generar_dataset(path="figures_clean", output_file="dataset.txt", save_contours_dir="contornos_prueba"):
    """Genera dataset de momentos de Hu desde imágenes preprocesadas.
    Si save_contours_dir se provee, guarda una imagen por archivo con el contorno
    dibujado en ese directorio, manteniendo estructura por clase.
    """
    if save_contours_dir:
        os.makedirs(save_contours_dir, exist_ok=True)

    with open(output_file, "w") as f:
        for clase in os.listdir(path):
            carpeta = os.path.join(path, clase)
            if not os.path.isdir(carpeta):
                continue
            if save_contours_dir:
                out_clase_dir = os.path.join(save_contours_dir, clase)
                os.makedirs(out_clase_dir, exist_ok=True)

            for archivo in os.listdir(carpeta):
                img_path = os.path.join(carpeta, archivo)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                hu, mask, cnt, vis = extraer_hu_y_contorno(img)
                if hu is None:
                    continue

                features_str = ",".join([str(float(v)) for v in hu])
                linea = f"{features_str},{clase}\n"
                f.write(linea)

                if save_contours_dir and vis is not None:
                    out_path = os.path.join(out_clase_dir, archivo)
                    cv2.imwrite(out_path, vis)

    print(f"Dataset guardado en {output_file}")
    if save_contours_dir:
        print(f"Imágenes con contornos guardadas en {save_contours_dir}")

if __name__ == "__main__":
    # Ejemplo: guarda dataset y también las imágenes con contorno en 'contours'
    generar_dataset(save_contours_dir="contours")
# ...existing code...