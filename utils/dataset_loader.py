import tensorflow as tf
from pathlib import Path


def load_datasets(config):

    dataset_config = config["dataset"]
    training_config = config["training"]


    dataset_path = Path(
        dataset_config["path"]
    )


    image_size = tuple(
        training_config["image_size"]
    )

    batch_size = training_config["batch_size"]


    train_path = dataset_path / dataset_config["train"]

    val_path = dataset_path / dataset_config.get(
        "validation",
        dataset_config.get("val")
    )

    test_path = dataset_path / dataset_config["test"]


    print("Dataset paths:")
    print("Train:", train_path)
    print("Validation:", val_path)
    print("Test:", test_path)


    for path in [
        train_path,
        val_path,
        test_path
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {path}"
            )


    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_path,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary"
    )


    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_path,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary"
    )


    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary"
    )


    autotune = tf.data.AUTOTUNE


    train_dataset = train_dataset.cache().prefetch(autotune)
    val_dataset = val_dataset.cache().prefetch(autotune)
    test_dataset = test_dataset.cache().prefetch(autotune)


    return (
        train_dataset,
        val_dataset,
        test_dataset
    )
