import tensorflow as tf

from utils.config_loader import load_config


config = load_config()


dataset_path = config["dataset"]["path"]


image_size = tuple(
    config["training"]["image_size"]
)

batch_size = config["training"]["batch_size"]


train_path = (
    f"{dataset_path}/"
    f"{config['dataset']['train']}"
)


train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=image_size,
    batch_size=batch_size,
    label_mode="binary"
)


print(
    "Dataset loaded:"
)

print(
    train_dataset
)
