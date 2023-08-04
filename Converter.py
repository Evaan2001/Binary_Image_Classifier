"""
I had to run my CNN on a Raspberry Pi 2 without a stable internet connection.
To make sure my program does not freeze, I decided to convert my TensorFlow model into a TensorFlow Lite model.
TensorFlow Lite is a lightweight version of TensorFlow that is designed for mobile and embedded devices. 
It provides a set of tools that enable on-device machine learning.

This code is available on TensorFlow's website. Here's the link – https://www.tensorflow.org/lite/models/convert/convert_models. 

"""

# import the relevant packages
import keras
import tensorflow as tf
import Essentials
import time

# Load the model
model_name = 'Custom_Loss.h5' # The path to the saved model
model = keras.models.load_model(model_name, compile=False)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
