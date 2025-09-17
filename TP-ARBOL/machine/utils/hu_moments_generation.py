# hu_moments_generation.py

import cv2
import os
import csv
import numpy as np

def generate_hu_moments_file(input_dir, output_file):
    """
    Genera un archivo con los 7 momentos de Hu para un conjunto de imágenes.
    
    Args:
        input_dir (str): Directorio de entrada con las subcarpetas de figuras.
        output_file (str): Ruta al archivo de salida donde se guardarán los resultados.
    """
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Escribe la cabecera del archivo CSV
        writer.writerow(['hu_1', 'hu_2', 'hu_3', 'hu_4', 'hu_5', 'hu_6', 'hu_7', 'class'])

        # Recorre cada clase (corazones, estrellas, etc.)
        for class_name in os.listdir(input_dir):
            class_path = os.path.join(input_dir, class_name)
            
            # Asegúrate de que solo procesas directorios
            if not os.path.isdir(class_path):
                continue
            
            print(f"Procesando clase: {class_name}")

            # Recorre cada imagen en la clase actual
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                
                # Lee la imagen y la convierte a escala de grises
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    print(f"Error: No se pudo leer la imagen {image_path}")
                    continue

                # Binariza la imagen para obtener la silueta
                # (150 es un umbral, puede que necesites ajustarlo)
                ret, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)

                # Busca los contornos en la imagen binarizada
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # Toma el contorno más grande (que probablemente sea la figura principal)
                    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
                    
                    # Calcula los momentos de la imagen
                    moments = cv2.moments(cnt)
                    
                    # Calcula los 7 momentos de Hu a partir de los momentos de la imagen
                    hu_moments = cv2.HuMoments(moments)
                    
                    # Usa log para normalizar los momentos de Hu y evitar rangos de valores muy grandes
                    log_hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

                    # Prepara la fila de datos para el CSV
                    row = [val[0] for val in log_hu_moments]
                    row.append(class_name)
                    
                    # Escribe la fila en el archivo
                    writer.writerow(row)