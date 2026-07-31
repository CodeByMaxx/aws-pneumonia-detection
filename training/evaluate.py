import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

from utils.config_loader import load_config



config = load_config()


model_path = config["model"]["path"]


model = tf.keras.models.load_model(
    model_path
)


print(
    "Model loaded:"
)

print(
    model_path
)


predictions = model.predict(
    test_dataset
)


print(
    classification_report(
        test_labels,
        predictions > 0.5
    )
)


print(
    confusion_matrix(
        test_labels,
        predictions > 0.5
    )
)
