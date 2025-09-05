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

def generar_dataset(path="figures_clean", output_file="dataset.txt"):
    """Genera dataset de momentos de Hu desde imágenes preprocesadas."""
    with open(output_file, "w") as f:
        for clase in os.listdir(path):
            carpeta = os.path.join(path, clase)
            for archivo in os.listdir(carpeta):
                img_path = os.path.join(carpeta, archivo)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                features = extraer_hu_moments(img)
                features_str = ",".join([str(v) for v in features])
                linea = f"{features_str},{clase}\n"
                f.write(linea)
    print(f"Dataset guardado en {output_file}")

if __name__ == "__main__":
    generar_dataset()
