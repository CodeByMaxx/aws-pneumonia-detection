from pathlib import Path
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
import numpy as np


IMG_SIZE = (224, 224)
BATCH_SIZE = 8


BASE_DIR = Path(__file__).resolve().parent.parent

TEST_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "chest_xray"
    / "test"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "pneumonia_model.keras"
)


def load_test_data():

    test_ds = tf.keras.utils.image_dataset_from_directory(
        TEST_PATH,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False
    )

    return test_ds


def evaluate_model():

    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    print("Loading test data...")

    test_ds = load_test_data()


    print("Running predictions...")

    predictions = model.predict(test_ds)


    y_pred = (
        predictions > 0.5
    ).astype(int).flatten()


    y_true = np.concatenate(
        [
            y.numpy()
            for x, y in test_ds
        ]
    ).astype(int).flatten()


    print("\nClassification Report:\n")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "NORMAL",
                "PNEUMONIA"
            ]
        )
    )


    print("\nConfusion Matrix:\n")

    print(
        confusion_matrix(
            y_true,
            y_pred
        )
    )


if __name__ == "__main__":

    evaluate_model()
