from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "agromind_mobilenetv2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METADATA_PATH = MODEL_DIR / "metadata.json"

IMAGE_SIZE = (224, 224)
TARGET_CLASSES = (
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
)
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

LABEL_MAP = {
    "Tomato___healthy": "healthy",
    "Tomato___Early_blight": "early_blight",
    "Tomato___Late_blight": "late_blight",
    "Tomato___Leaf_Mold": "leaf_mold",
}

# Hackathon risk proxies, not lesion-area measurements. Keep this distinction in
# downstream product copy until a segmentation/severity model exists.
SEVERITY_MAP = {
    "healthy": 0.0,
    "leaf_mold": 0.55,
    "early_blight": 0.70,
    "late_blight": 0.90,
}

