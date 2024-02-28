import tensorflow as tf
from tensorflow.keras import layers

def separator(input_size, n_channel, n_class, **ignore):
  inputs = tf.keras.Input(shape = (input_size, input_size, n_channel))
  dense = lambda x, name: layers.Conv2D(filters = 32, kernel_size = 1, use_bias = True, name = name, activation = "relu")(x)
  x = dense(inputs, "l1")
  x = dense(x, "l2")
  x = layers.Conv2D(filters = 16, kernel_size = 1, use_bias = True, name = "latent_last", activation = "relu")(x)
  outputs = layers.Conv2D(filters = n_class, kernel_size = 1, use_bias = True, name = "head", activation = "softmax")(x)
  return tf.keras.Model(inputs, outputs)