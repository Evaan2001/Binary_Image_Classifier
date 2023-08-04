"""
File that builds and trains a CNN for binary image classification.
"""

# Import packages

import matplotlib.pyplot as plt
import os

plt.style.use('classic')

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

# Now import my Essentials.py file that has a bunch of helper functions

import Essentials
###################################

# Set class names to label the training/testing data
rock = "sunstone"
not_rock = "not_" + rock

# Now we'll create a variable that stores the details of the directory where training data is located
# The way I have set up the Essentials file, we need to follow the following format:
# DIRECTORY_DATA = [path to Directory, num of positive cases, no of negative cases, image height, image width]
DIRECTORY_NAME = "RL040420" #Name of the folder that has all the training images

# Now we will retrieve all of the images. I stored them as a compressed numpy file (npz file) for faster loading & processing
# Check out CreateData.py to see the code and for more information on this process 
color_data, grayscale_data = Essentials.getCompressedData(DIRECTORY_NAME) 

# A note on data organization. I use x to denote images and y for corresponding labels.
# So suppose the 50th image contained in x_training is of a sunstone rock. So the 50th value of y_training will be 1
# Similarly, suppose the 51st image contained in x_training is NOT of a sunstone, the 51st value of y_training will be 0

# All of our pixel brightness values are integers between 0-255. To ensure we are treating each pixel fairly, 
# we will normalize all of the images so that their brightness values are decimals between 0 & 1. We can do 
# this by simply dividing the image pixel values by 255
normalization_Factor = 255.0

# Retrieve and normalize the training data
x_training = color_data["x_training"] / normalization_Factor
y_training = color_data["y_training"]

# Retrieve and normalize the cross-validation data
x_cv = color_data["x_cv"] / normalization_Factor
y_cv = color_data["y_cv"]

# Retrieve and normalize the testing data
x_test = color_data["x_test"] / normalization_Factor
y_test = color_data["y_test"]

# Now we will set up the model architecture for our CNN
INPUT_SHAPE = (112, 112, 3) # Shape of the color images we have

model = Sequential() # Our CNN is a linear neural network (no RNNs) 
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE)) # 1st layer is convolutional layer with 32 3*3 kernels, the output will be 32 feature maps
model.add(Activation('relu')) # Now we will run the relu function on the output of each convolution 
model.add(MaxPooling2D(pool_size=(2, 2))) # we will choose the most prominent (aka, max value) feature from each 2*2 square; 
                                          # this will halve the dimensions of the feature maps

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform')) # 2nd convolutional layer again with 32 3*3 kernels, output will be 32 feature maps 
model.add(Activation('relu')) # Applying the relu function on the output of each convolution again
model.add(MaxPooling2D(pool_size=(2, 2))) # Maxpooling again as well

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform')) # 3rd convolutional layer with 64 3*3 kernels, output will be 64 feature maps
model.add(Activation('relu')) # Applying the relu function on the output of each convolution again
model.add(MaxPooling2D(pool_size=(2, 2))) # Maxpooling again as well

model.add(Flatten()) # Flatten the model, i.e., take the 64 output feature maps from the last layer & rearrange them as a 1-D array
model.add(Dense(64)) # Add a dense (aka, fully connected) layer with 64 nodes. Think of this as taking the most prominent feature from each of the 64 feature maps
model.add(Activation('relu')) # Apply relu again
model.add(Dropout(0.5)) # randomly drop out half of the 64 nodes in the previous layer to prevent overfitting

model.add(Dense(1)) # Take a weighted sum of the remaining 32 nodes
model.add(Activation('sigmoid')) # Apply the sigmoid activation function to calculate the probability of an image having a sunstone

# Our data is heavily unbalanced. We have only 334 samples of sunstone images but 3074 not_sunstone images (that's a 10x difference!)
# If we train our model with the data set, it will hardly learn how to identify sunstones as all the information it's getting is primarily for not_sunstone
# To solve this, we will set class weights. We want the model to get an equal amount of info on sunstones & not_sunstones to prevent overfitting.
# For each sunstone image, we have 3074/334 ≈ 9.2 images. So we'll instruct the model to consider each sunstone image as approximately 9.2 non_sunstone images.
# All we need to do is set class weights.
weights = [1, 9.2] # The weight for class 0 (aka, not_sunstone) is 1 and the weight for class 1 (aka, not_sunstone) is 9.2

# We will train the model over 20 epochs. In case later epochs cause overfitting, we will save the weights after each epoch as checkpoints
checkpoint_path = "training/cp-{epoch:02d}.ckpt" # where we are storing the checkpoints
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, # At the location specified in checkpoint_path
                                                 save_weights_only=True,   # Save the model weights
                                                 save_freq='epoch',        # After each epoch
                                                 verbose=1)

# Now compile the model (set up the matrices to store model weights)
model.compile(loss=keras.losses.binary_crossentropy, # Using the binary cross-entropy function to calculate loss (error)
              loss_weights=weights, # Using our custom model weights (specified in line 82)
              optimizer='adam',     # Using the adam optimizer with gradient descent
              metrics=['accuracy', 'FalseNegatives']) # And focusing on accuracy and false-negative rate

# Finally train the model
history = model.fit(x_training, # training images
                    y_training, # training labels
                    batch_size=64, # adjust the weights after 64 images have been processed
                    verbose=1,
                    epochs=20, # go through the training data set 20 times
                    validation_data=(x_cv, y_cv), # cross-validate using this data
                    shuffle=False, # Don't shuffle (randomize) the data as functions in the Essentials.py file already do this
                    callbacks=[cp_callback] # and save checkpoints
                    )

# Now we'll plot the loss data
loss = history.history['loss']         # Retrieve the training loss recorded after each epoch
val_loss = history.history['val_loss'] # Retrieve the cross-validation loss recorded after each epoch
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')       # Plot the training loss 
plt.plot(epochs, val_loss, 'r', label='Validation loss') # Plot the cross-validation loss
plt.title('Training and validation loss') # Title of the plot
plt.xlabel('Epochs') # With epoch number on the x-axis
plt.ylabel('Loss')   # And loss values on the y-axis
plt.legend() # Come up with a legend for the graph
plt.show()   # Finally display the plot

# Now we'll plot the accuracy data
acc = history.history['accuracy']         # Retrieve the training accuracy recorded after each epoch
val_acc = history.history['val_accuracy'] # Retrieve the cross-validation accuracy recorded after each epoch
plt.plot(epochs, acc, 'y', label='Training acc')       # Plot the training accuracy 
plt.plot(epochs, val_acc, 'r', label='Validation acc') # Plot the cross-validation accuracy
plt.title('Training and validation Acc') # Title of the plot
plt.xlabel('Epochs')   # With epoch number on the x-axis
plt.ylabel('Accuracy') # And loss values on the y-axis
plt.legend() # Come up with a legend for the graph
plt.show()   # Finally display the plot

false_negatives = history.history['false_negatives']         # Retrieve the training false-negative rate recorded after each epoch
val_false_negatives = history.history['val_false_negatives'] # Retrieve the cross-validation false-negative rate recorded after each epoch
plt.plot(epochs, false_negatives, 'y', label='Training FalseNegatives')       # Plot the training false-negative rate 
plt.plot(epochs, val_false_negatives, 'r', label='Validation FalseNegatives') # Plot the cross-validation false-negative rate
plt.title('Training and validation FalseNegatives') # Title of the plot
plt.xlabel('Epochs')         # With epoch number on the x-axis
plt.ylabel('FalseNegatives') # And false-negative rate on the y-axis
plt.legend() # Come up with a legend for the graph
plt.show()   # Finally display the plot

# Finally we will use the model on new data – the testing data (x_test & y_test)
prediction = model.predict(x_test)      # Predict the labels for the images in x_test
Essentials.accuracy(prediction, y_test) # Check if they match with y_test (the respective ground truth labels)
