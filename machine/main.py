import os
from machine.utils.hu_moments_generation import generate_hu_moments_file
from machine.utils.testing_model import load_and_test
from machine.utils.training_model import train_model

# 📂 Ruta base donde está este archivo (main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📂 Carpeta donde se guardarán los archivos generados
GENERATED_DIR = os.path.join(BASE_DIR, "generated-files")
os.makedirs(GENERATED_DIR, exist_ok=True)

# 📄 Ruta absoluta al archivo CSV
OUTPUT_FILE = os.path.join(GENERATED_DIR, "shapes-hu-moments.csv")

# 🚀 Ejecutar flujo completo
generate_hu_moments_file(OUTPUT_FILE)   # <- le pasamos la ruta absoluta
model = train_model(OUTPUT_FILE)
load_and_test(model)

