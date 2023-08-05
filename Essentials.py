"""
This file contains the helper functions I used frequently.
The first 2 functions are about retrieving image data.
Then we have a function that measures the model performance
Lastly, we have 3 functions that split the data into training, cross-validation, and testing sets
"""

# Import relevant packages
import cv2
import numpy as np
import glob


def getCompressedData(directory):
    """
    I converted the original images into a compressed numpy array file (check out CreateData.py for more info).
    This function accesses those compressed arrays and returns the images to be used for training & testing.

    Input:
        directory – path to the directory containing the compressed numpy arrays
    Outputs:
        color_data     – an array containing color images grouped into training, cross-validation, & testing sets
        grayscale_data – an array containing grayscale images grouped into training, cross-validation, & testing sets
    """
    file_extension = ".npz" 

    color_file_name = "color_" + directory + file_extension # That's how I named the compressed array file containing color images
    color_data = np.load(color_file_name)                   # Load the file

    grayscale_file_name = "grayscale_" + directory + file_extension # That's how I named the compressed array file containing grayscale images
    grayscale_data = np.load(grayscale_file_name)                   # Load the file

    return color_data, grayscale_data # Return the retrieved files

def getData(directory, no_of_sunstones, no_of_not_sunstones, image_length, image_breadth):
    """
    This function accesses the images used for training & testing the model and returns them as a giant numpy array
    Inputs:
        directory – path to the directory containing the images
        no_of_sunstones – number of images that have a sunstone
        no_of_not_sunstones – number of images that don't have a sunstone
        image_length – the width of an image
        image_breadth – the height of an image
    Outputs:
        m – total number of images
        X_Grayscale – numpy array containing grayscale images
        X_Color – numpy array containing color images 
        Y – corresponding image label. So if Y[30] == 1, the 30th image in X_Grayscale & X_Color is of a sunstone)
                                       Analogously, if Y[35] == 0, the 35th image in X_Grayscale & X_Color is of a not_sunstone)
    """
    m = no_of_sunstones + no_of_not_sunstones

    # Format of an image
    num_Grayscale_Channels = 1
    num_RGB_Channels = 3

    # Creating array X and initializing with all ones
    X_Grayscale_Dimensions = (m, image_length, image_breadth)
    X_Grayscale = np.zeros(X_Grayscale_Dimensions)  # This variable stores all the training images

    X_Color_Dimensions = (m, image_length, image_breadth, num_RGB_Channels)
    X_Color = np.zeros(X_Color_Dimensions)  # This variable stores all the training images

    # Time to load all the images
    # We will do this in 2 steps. First, we'll get all the Sunstone images
    # Then we'll retrieve all the Not_Sunstone images
    i = 0  # to use as an array indexer
    location = '/Users/evaanahmed/Desktop/Sunstone Sorter/'  # The location to the directory containing Sunstone images
    status = ['Sunstones', 'Not_Sunstones']
    for val in status:
        for filename in glob.glob(location + directory + '/' + val + '/*.jpg'):
            img_Grayscale = cv2.imread(filename,
                                       cv2.IMREAD_GRAYSCALE)  # retreive an image and store it as a numpy array
            img_Color = cv2.imread(filename)
            X_Grayscale[i, :, :] = img_Grayscale
            X_Color[i, :, :] = img_Color
            i = i + 1

    # Now, we'll create an array Y of size m X 1 (m is the total number of training images) that:
    # stores 1 if the corresponding index in X has a sunstone image
    # stores 0 if the corresponding index in X does not have a sunstone image
    Y = np.append(np.ones(no_of_sunstones), np.zeros(no_of_not_sunstones))

    # Our data is extremely ordered right now: it has all the sunstones together followed by all not_sunstones
    # To make this training data less biased, let's simultaneously shuffle X and Y
    shuffler = np.random.permutation(m)
    X_Grayscale = X_Grayscale[shuffler, :]
    X_Color = X_Color[shuffler, :]
    Y = Y[shuffler]

    # Finally, we'll transpose Y to make it a column vector (to make life easier when dealing with matrix multiplications)
    Y = Y[..., None]

    return m, X_Grayscale, X_Color, Y


def accuracy(h, y):
    """ 
    Function to test the accuracy of a model's predictions.
    Input: 
        h -> model hypothesis (aka, predictions) 
        y -> Actual/Real Values
    Output: 
        No return values; the following get printed
        The precision of hypothesis h, Amount of sunstones wasted (more or less false negatives), Overall accuracy of the hypothesis h
    """

    m = len(y)        # get the totla no of images
    true_positives = len(y[y == 1]) # get the number of sunstone images
    true_negatives = len(y[y == 0]) # get the number of not_sunstone images

    hypothesis_positives = len(h[h >= 0.5]) # get the number of positive classifications in h (so no instances where hypothesis probability ≥ 0.5)

    false_positives = 0
    false_negatives = 0

    for i in range(len(h)):
        if h[i] >= 0.5 and y[i] == 0: # check if the current classification is a false positive
            false_positives += 1
        elif h[i] < 0.5 and y[i] == 1: # check if the current classification is a false negative
            false_negatives += 1

    total_errors = false_positives + false_negatives

    precision = (hypothesis_positives - false_positives) / hypothesis_positives # calculate the precision of the model
    precision *= 100

    wastage = false_negatives * 100 / true_positives # calculate the false_negatives rate

    overall_accuracy = 100 - (total_errors * 100 / m) # calculate the overall accuracy

    # Now display every metric
    print("Precision of output =  " + str(precision) + "%")
    print("Sunstone Wastage =  " + str(wastage) + "%")
    print("Overall accuracy =  " + str(overall_accuracy) + "%", end = '\n\n')

def splitDataTraining(x, y):
    """
    This function takes in the complete set of images and image labels, then separates out the first 60% of data for training
    """
    m = np.size(y)          # get the total no of images
    training_start = 0      # starting index for separating out the data
    cv_start = int(0.6 * m) # ending index

    # Now get the data between those indices 
    x_training = x[training_start:cv_start] 
    y_training = y[training_start:cv_start] 

    # Return the acquired data
    return x_training, y_training

def splitDataCV(x, y):
    """
    This function takes in the complete set of images and image labels, then separates out 20% of data for cross-validation
    The 1st 60% of data is for training
    The next 20% is for cross-validation
    The last 20% is for testing
    """
    m = np.size(y)            # get the total no of images
    cv_start = int(0.6 * m)   # starting index for separating out the data
    test_start = int(0.8 * m) # ending index

    # Now get the data between those indices 
    x_cv = x[cv_start:test_start]
    y_cv = y[cv_start:test_start]

    # Return the acquired data
    return x_cv, y_cv

def splitDataTesting(x, y):
    """
    This function takes in the complete set of images and image labels, then separates out the last 20% of data for testing
    """
    m = np.size(y)            # get the total no of images
    test_start = int(0.8 * m) # starting index for separating out the data

    # Now get the data from the starting index till the end
    x_test = x[test_start:]
    y_test = y[test_start:]

    # Return the acquired data
    return x_test, y_test
