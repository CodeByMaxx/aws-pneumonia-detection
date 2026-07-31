import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

import tensorflow as tf

from utils.config_loader import load_config
from utils.dataset_loader import load_datasets

config = load_config()

train_dataset, val_dataset, test_dataset = load_datasets(
    config
)

image_size = tuple(
    config["training"]["image_size"]
)

epochs = config["training"]["epochs"]

batch_size = config["training"]["batch_size"]

learning_rate = config["training"]["learning_rate"]

model_path = config["model"]["path"]



base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(
        image_size[0],
        image_size[1],
        3
    ),
    include_top=False,
    weights="imagenet"
)


base_model.trainable = False


model = tf.keras.Sequential([

    base_model,

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate
    ),

    loss="binary_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("Starting training")


model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=epochs

)


model.save(model_path)


print(
    f"Saved model: {model_path}"
)
