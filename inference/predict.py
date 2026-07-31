import sys

import tensorflow as tf

import numpy as np

from tensorflow.keras.preprocessing import image


from utils.config_loader import load_config



config = load_config()



model = tf.keras.models.load_model(

    config["model"]["path"]

)


img_path = sys.argv[1]


size = tuple(

    config["training"]["image_size"]

)



img = image.load_img(

    img_path,

    target_size=size

)


img_array = image.img_to_array(
    img
)


img_array = np.expand_dims(

    img_array,

    axis=0

)


prediction = model.predict(
    img_array
)[0][0]



threshold = config["inference"]["confidence_threshold"]



if prediction >= threshold:

    result = "PNEUMONIA"

else:

    result = "NORMAL"



print(
    "Prediction:"
)

print(
    result
)


print(
    f"Confidence: {prediction:.2%}"
)
