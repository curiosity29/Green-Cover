import numpy as np
from Postprocess import amplify
import tensorflow as tf
def predict_adapter(batch, model):
  ## dilated predict
  predict = model.predict(tf.concat([batch,batch], axis = -1), verbose = 0)[:, 1, ...]

  predict = np.array([amplify(predict_, strength = 3) for predict_ in predict])
  predict = np.argmax(predict, axis = -1)
  predict = predict[..., np.newaxis]

  return predict

