from pathlib import Path
import sys
import tensorflow as tf
import numpy as np


IMG_SIZE = (224, 224)


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "pneumonia_model.keras"
)


def load_model():

    print("Loading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


def preprocess_image(image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = tf.keras.utils.img_to_array(
        image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


def predict(image_path):

    model = load_model()

    image = preprocess_image(
        image_path
    )

    prediction = model.predict(
        image
    )[0][0]


    if prediction >= 0.5:

        result = "PNEUMONIA"
        confidence = prediction

    else:

        result = "NORMAL"
        confidence = 1 - prediction


    print("\nPrediction:")
    print(result)

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python predict.py <image_path>"
        )

        sys.exit(1)


    image_path = sys.argv[1]

    predict(image_path)
