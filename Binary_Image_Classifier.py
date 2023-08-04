import matplotlib.pyplot as plt
import os

plt.style.use('classic')

#import packages
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

#now import my Essentials.py file that has a bunch of helper functions

import Essentials
###################################

#set class names to label the training/testing data
rock = "sunstone"
not_rock = "not_" + rock

# Now we'll create a variable that stores the details of the directory where training data is located
# The way I have set up the Essentials file, we need to follow the following format:
# DIRECTORY_DATA = [path to Directory, num of positive cases, no of negative cases, image height, image width]
DIRECTORY_NAME = "RL040420" #Name of the folder that has all the training images
num_positive_cases = 335 #Easy to get these numbers with MacOS/Windows but I'm sure this can be easily calculated with a Python script
num_negative_cases = 3080
DIRECTORY_DATA = [DIRECTORY_NAME", 335, 3080, 112, 112]

color_data, grayscale_data = Essentials.getCompressedData(DIRECTORY) #retrieve the images from

normalization_Factor = 255.0

x_training = color_data["x_training"] / normalization_Factor
y_training = color_data["y_training"]

x_cv = color_data["x_cv"] / normalization_Factor
y_cv = color_data["y_cv"]

x_test = color_data["x_test"] / normalization_Factor
y_test = color_data["y_test"]

INPUT_SHAPE = (112, 112, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

weights = [1/3079, 1/334]

checkpoint_path = "training_3/cp-{epoch:02d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 save_freq='epoch',
                                                 verbose=1)

model.compile(loss=keras.losses.binary_crossentropy,
              loss_weights=weights,
              optimizer='adam',
              metrics=['accuracy', 'FalseNegatives'])

history = model.fit(x_training,
                    y_training,
                    batch_size=64,
                    verbose=1,
                    epochs=20,
                    validation_data=(x_cv, y_cv),
                    shuffle=False,
                    callbacks=[cp_callback]
                    )

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

false_negatives = history.history['false_negatives']
val_false_negatives = history.history['val_false_negatives']
plt.plot(epochs, false_negatives, 'y', label='Training FalseNegatives')
plt.plot(epochs, val_false_negatives, 'r', label='Validation FalseNegatives')
plt.title('Training and validation FalseNegatives')
plt.xlabel('Epochs')
plt.ylabel('FalseNegatives')
plt.legend()
plt.show()

h_test = model.predict(x_test)
Essentials.accuracy(h_test, y_test)
