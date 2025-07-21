import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("Metal support:", len(tf.config.list_physical_devices('GPU')) > 0)