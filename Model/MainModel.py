from separator import separator
from U2Net import U2Net_augment
import Configs
import tensorflow as tf
from tensorflow.keras import layers

def U2Net_dilated(pretrainable = False, dilation_size = 9, U2Net_version = 2, input_size = 512, n_channel = 4, n_class = 4):
  args = dict(input_size=input_size, n_channel=n_channel, n_class=n_class)
  aug_model = separator(**args)
  aug_model.load_weights("/content/drive/MyDrive/ColabShared/Checkpoint/Multiclass_segmentation/PoolsAT_separator_17_02_v2_last.weights.h5")
  aug_model.trainable = pretrainable
  # aug_model = tf.keras.Model(inputs= aug_model.input, outputs = aug_model.get_layer("latent_last").output)

  inputs = tf.keras.Input(shape = (input_size, input_size, n_channel * 2))

  raw_inputs = inputs[..., :n_channel]
  coarse_map = aug_model(raw_inputs) # take the raw input in the first half
  # use base model on augmented input
  dilated_inputs = inputs[..., n_channel:]

  dilated_outputs = U2Net_augment(dilated_inputs, coarse_map = coarse_map, version = U2Net_version, check_ignore = False, **args) # take the augmented input in the second half
  dilated_main_output = dilated_outputs[:, 0, ...]

  ## normal label from dilated label
  outputs = layers.Conv2D(filters = n_class, kernel_size = dilation_size, padding = "same", activation = "relu")(dilated_main_output)
  outputs = layers.Conv2D(filters = n_class, kernel_size = dilation_size, padding = "same", activation = "sigmoid")(outputs)
  outputs = outputs[:, tf.newaxis, ...]
  outputs = layers.Concatenate(axis = 1)([outputs, dilated_outputs])
  # outputs stack at axis 1: label predict, dilated label predict, dilated sides x6
  return tf.keras.Model(inputs, outputs)

