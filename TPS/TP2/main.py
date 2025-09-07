from preprocess import preprocess_dataset
from hu_moments import generar_dataset
from model import entrenar_modelo, reconocimiento_en_vivo

if __name__ == "__main__":
    print("=== Paso 1: Preprocesando imágenes ===")
    preprocess_dataset("figures", "figures_clean")

    print("\n=== Paso 2: Generando dataset de Hu Moments ===")
    generar_dataset("figures_clean", "dataset.txt")

    print("\n=== Paso 3: Entrenando modelo ===")
    modelo = entrenar_modelo()

    print("\n=== Paso 4: Reconocimiento en vivo ===")
    reconocimiento_en_vivo(modelo)

    
