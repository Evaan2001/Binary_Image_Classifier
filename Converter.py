import keras
import tensorflow as tf
import Essentials
import time

print("Loading Model")
model = keras.models.load_model('Custom_Loss.h5', compile=False)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)