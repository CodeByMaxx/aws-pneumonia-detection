import tensorflow as tf

print("GPUs:")
print(tf.config.list_physical_devices("GPU"))

with tf.device("/GPU:0"):
    a = tf.random.normal((5000, 5000))
    b = tf.random.normal((5000, 5000))
    c = tf.matmul(a, b)

print("GPU calculation successful")
print(c.shape)
