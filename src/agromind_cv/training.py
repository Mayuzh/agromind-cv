import argparse
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from .config import (
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    PROCESSED_DATA_DIR,
)

AUTOTUNE = tf.data.AUTOTUNE


def configure_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def load_datasets(data_dir: Path, batch_size: int, seed: int):
    common = {"image_size": IMAGE_SIZE, "batch_size": batch_size, "label_mode": "int"}
    train = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train", shuffle=True, seed=seed, **common
    )
    validation = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val", shuffle=False, **common
    )
    test = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test", shuffle=False, **common
    )
    class_names = train.class_names
    if validation.class_names != class_names or test.class_names != class_names:
        raise ValueError("Train, validation, and test class folders do not match.")
    return (
        train.prefetch(AUTOTUNE),
        validation.prefetch(AUTOTUNE),
        test.prefetch(AUTOTUNE),
        class_names,
    )


def class_weights(train_ds, class_count: int) -> dict[int, float]:
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in train_ds])
    classes = np.arange(class_count)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {
        int(label): float(weight) for label, weight in zip(classes, weights, strict=True)
    }


def build_model(class_count: int):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(class_count, activation="softmax", name="class")(x)
    return tf.keras.Model(inputs, outputs), base


def callbacks() -> list[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_loss", mode="min", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.3, patience=2, min_lr=1e-7
        ),
    ]


def save_learning_curves(histories: list[tf.keras.callbacks.History]) -> None:
    accuracy = sum((history.history.get("accuracy", []) for history in histories), [])
    val_accuracy = sum((history.history.get("val_accuracy", []) for history in histories), [])
    loss = sum((history.history.get("loss", []) for history in histories), [])
    val_loss = sum((history.history.get("val_loss", []) for history in histories), [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(accuracy, label="train")
    axes[0].plot(val_accuracy, label="validation")
    axes[0].set(title="Accuracy", xlabel="Epoch")
    axes[0].legend()
    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="validation")
    axes[1].set(title="Loss", xlabel="Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "learning_curves.png", dpi=160)
    plt.close(fig)


def evaluate(model, test_ds, class_names: list[str]) -> dict[str, object]:
    true_labels = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = model.predict(test_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    labels = list(range(len(class_names)))
    report = classification_report(
        true_labels,
        predictions,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predictions, labels=labels)
    (MODEL_DIR / "classification_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savetxt(MODEL_DIR / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            axis.text(
                column,
                row,
                matrix[row, column],
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


def train(args: argparse.Namespace) -> None:
    configure_runtime(args.seed)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.data.resolve(), args.batch_size, args.seed
    )
    CLASS_NAMES_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    weights = class_weights(train_ds, len(class_names))
    model, base = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.head_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"TensorFlow {tf.__version__}; GPUs: {tf.config.list_physical_devices('GPU')}")
    print("Training classifier head...")
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=weights,
        callbacks=callbacks(),
    )

    print("Fine-tuning MobileNetV2 tail...")
    base.trainable = True
    for layer in base.layers[:-args.unfreeze_layers]:
        layer.trainable = False
    # Keep batch-normalization statistics frozen for stable small-dataset tuning.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.fine_tune_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        class_weight=weights,
        callbacks=callbacks(),
    )
    save_learning_curves([head_history, fine_history])

    best_model = tf.keras.models.load_model(MODEL_PATH)
    test_loss, test_accuracy = best_model.evaluate(test_ds, verbose=1)
    report = evaluate(best_model, test_ds, class_names)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tensorflow_version": tf.__version__,
        "image_size": list(IMAGE_SIZE),
        "class_names": class_names,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "severity_note": (
            "Severity values are fixed disease-risk proxies, not measured lesion severity."
        ),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved best model to {MODEL_PATH}")
    print(f"Test accuracy: {test_accuracy:.4f}; macro F1: {metadata['macro_f1']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AgroMind MobileNetV2 classifier.")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--unfreeze-layers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
