import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

from pathlib import Path
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy("float32")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(
            gpu,
            True
        )

    print("GPU memory growth enabled")


from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


IMG_SIZE = (160, 160)
BATCH_SIZE = 8
EPOCHS = 10


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "raw" / "chest_xray" / "train"
VAL_PATH = BASE_DIR / "data" / "raw" / "chest_xray" / "val"

MODEL_PATH = BASE_DIR / "models" / "pneumonia_model.keras"


def load_dataset(path):

    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )


def build_model():

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(160,160,3)
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    return model


if __name__ == "__main__":

    print("Loading dataset...")

    train_ds = load_dataset(TRAIN_PATH)
    val_ds = load_dataset(VAL_PATH)


    model = build_model()

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    model.summary()


    print("Starting training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )


    MODEL_PATH.parent.mkdir(
        exist_ok=True
    )

    model.save(MODEL_PATH)

    print(
        f"Model saved to {MODEL_PATH}"
    )
