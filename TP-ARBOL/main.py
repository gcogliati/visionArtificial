import os
from machine.utils.hu_moments_generation import generate_hu_moments_file
from machine.utils.testing_model import load_and_test
from machine.utils.training_model import train_model

# 📂 Ruta base donde está este archivo (main.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 📂 Carpetas de datos
TRAIN_DIR = os.path.join(BASE_DIR, "data", "raw", "train")
TEST_DIR = os.path.join(BASE_DIR, "data", "raw", "test")

# 📂 Carpeta donde se guardarán los archivos generados
GENERATED_DIR = os.path.join(BASE_DIR, "generated-files")
os.makedirs(GENERATED_DIR, exist_ok=True)

# 📄 Rutas a CSVs generados
TRAIN_CSV = os.path.join(GENERATED_DIR, "train-hu-moments.csv")
TEST_CSV = os.path.join(GENERATED_DIR, "test-hu-moments.csv")

# 🤖 Modelos
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "decision_tree_hu.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

# 📊 Reportes
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)
CONF_MAT_PATH = os.path.join(FIG_DIR, "confusion_matrix.png")

def main():
    print("==> 1) Generando CSV de Momentos de Hu (TRAIN)")
    generate_hu_moments_file(output_csv=TRAIN_CSV, input_dir=TRAIN_DIR)

    print("==> 2) Generando CSV de Momentos de Hu (TEST)")
    generate_hu_moments_file(output_csv=TEST_CSV, input_dir=TEST_DIR)

    print("==> 3) Entrenando modelo (árbol de decisión)")
    model, encoder = train_model(
        features_csv_path=TRAIN_CSV,
        model_path=MODEL_PATH,
        encoder_path=ENCODER_PATH,
        cv_folds=5
    )

    print("==> 4) Evaluando en TEST y guardando matriz de confusión")
    load_and_test(
        model_path=MODEL_PATH,
        encoder_path=ENCODER_PATH,
        test_csv=TEST_CSV,
        fig_path=CONF_MAT_PATH
    )

    print("\n✅ Listo. ¡Pipeline completo ejecutado!")
    print(f"Modelo:        {MODEL_PATH}")
    print(f"LabelEncoder:  {ENCODER_PATH}")
    print(f"Confusion:     {CONF_MAT_PATH}")
    print(f"Train CSV:     {TRAIN_CSV}")
    print(f"Test  CSV:     {TEST_CSV}")

if __name__ == "__main__":
    main()
