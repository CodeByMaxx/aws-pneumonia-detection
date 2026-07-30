from pathlib import Path
import tensorflow as tf


IMG_SIZE = (224, 224)
BATCH_SIZE = 32


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data" / "raw" / "chest_xray" / "train"


def load_dataset(path):

    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    return dataset


if __name__ == "__main__":

    print(f"Dataset path: {TRAIN_PATH}")

    train_ds = load_dataset(TRAIN_PATH)

    print(train_ds)
