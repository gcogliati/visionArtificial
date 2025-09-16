import cv2
import csv
import glob
import numpy
import math
import os

# Escribo los valores de los momentos de Hu en el archivo
def write_hu_moments(label, writer, base_path="./shapes"):
    path = os.path.join(base_path, label, "*")
    files = glob.glob(path)
    if not files:
        print(f"[WARN] No se encontraron imágenes en {path}")
        return
    
    for file in files:
        hu = hu_moments_of_file(file)
        flattened = hu.ravel()
        row = numpy.append(flattened, label)
        writer.writerow(row)

def generate_hu_moments_file(output_path, base_path="./shapes"):
    with open(output_path, 'w', newline='') as file:
        writer = csv.writer(file)
        # Escribir encabezado
        header = [f"hu{i+1}" for i in range(7)] + ["label"]
        writer.writerow(header)

        # Escribir datos de cada clase
        write_hu_moments("5-point-star", writer, base_path)
        write_hu_moments("rectangle", writer, base_path)
        write_hu_moments("triangle", writer, base_path)

# Encargada de generar los momentos de Hu para las imagenes
def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    if image is None:
        raise FileNotFoundError(f"No pude abrir la imagen {filename}")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 67, 2)

    # Invertir para que la figura quede en blanco
    bin = 255 - bin
    kernel = numpy.ones((3, 3), numpy.uint8)
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, _ = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No se encontraron contornos en {filename}")

    shape_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(shape_contour)
    huMoments = cv2.HuMoments(moments)

    # Log scale
    for i in range(7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
    return huMoments
