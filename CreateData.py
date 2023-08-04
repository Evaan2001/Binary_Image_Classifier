"""
Loading images to train multiple models was very time-consuming.
So I chose to first load the images into numpy arrays and then store
them in a single compressed "npz" file. This is a compressed file format 
that allows us to save and load multiple arrays efficiently.
"""

import Essentials # Import my helper functions
import numpy as np

# Now set the parameters to send to Essentials.getData
DIRECTORY_NAME = "RL040420" # Name of the folder that has all the training images
num_positive_cases = 334 
num_negative_cases = 3079
IMAGE_LENGTH = 112
IMAGE_BREADTH = 112

# Now call Essentials.getData to get the images as numpy arrays
m, X_Grayscale, X_Color, Y = Essentials.getData(DIRECTORY_NAME, num_positive_cases, num_negative_cases, IMAGE_LENGTH, IMAGE_BREADTH)
# m <- total no of images
# X_Grayscale <- images in grayscale
# X_Color <- images in color
# Y <- truth labels (0 means the picture is not of a sunstone, and 1 means the picture represents a sunstone

# Now that we loaded all the images, we need to split them into training, cross-validation, and testing sets

# Let's first work with the color images
x_training, y_training = Essentials.splitDataTraining(X_Color, Y) # Use the first 60% of color images for training
x_cv, y_cv = Essentials.splitDataCV(X_Color, Y) # Use the next 20% of color images for cross-validation
x_test, y_test = Essentials.splitDataTesting(X_Color, Y) # Use the last 20% of color images for testing

# And now we just need to save the color images data in a npz file
np.savez_compressed('color_RL040420.npz', x_training=x_training,
                    y_training=y_training,
                    x_cv=x_cv,
                    y_cv=y_cv,
                    x_test=x_test,
                    y_test=y_test)

# We are done with color images
# Now let's split and save grayscale images
x_training, y_training = Essentials.splitDataTraining(X_Grayscale, Y) # Use the first 60% of grayscale images for training
x_cv, y_cv = Essentials.splitDataCV(X_Grayscale, Y) # Use the next 20% of grayscale images for cross-validation
x_test, y_test = Essentials.splitDataTesting(X_Grayscale, Y) # Use the last 20% of grayscale images for testing

# And now we just need to save the grayscale images data in a npz file
np.savez_compressed('grayscale_RL040420.npz', x_training=x_training,
                    y_training=y_training,
                    x_cv=x_cv,
                    y_cv=y_cv,
                    x_test=x_test,
                    y_test=y_test)
