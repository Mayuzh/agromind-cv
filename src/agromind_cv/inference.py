import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from .config import CLASS_NAMES_PATH, IMAGE_SIZE, LABEL_MAP, MODEL_PATH, SEVERITY_MAP


class ModelUnavailableError(RuntimeError):
    pass


def preprocess_image(source: str | Path | bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(source) if isinstance(source, bytes) else source).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("The supplied file is not a readable image.") from error
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)


@lru_cache(maxsize=1)
def load_artifacts() -> tuple[tf.keras.Model, list[str]]:
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise ModelUnavailableError("Model artifacts are missing. Run training first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    if model.output_shape[-1] != len(class_names):
        raise ModelUnavailableError("Model output and class_names.json do not agree.")
    return model, class_names


def predict_image(source: str | Path | bytes) -> dict[str, object]:
    model, class_names = load_artifacts()
    probabilities = model.predict(preprocess_image(source), verbose=0)[0]
    index = int(np.argmax(probabilities))
    raw_label = class_names[index]
    prediction = LABEL_MAP.get(raw_label, raw_label)
    confidence = float(probabilities[index])
    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "severity": SEVERITY_MAP.get(prediction, 0.0),
    }

