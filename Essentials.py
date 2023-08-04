import cv2
import numpy as np
import glob

def getCompressedData(directory):

    file_format = ".npz"

    color_file_name = "color_" + directory + file_format
    color_data = np.load(color_file_name)

    grayscale_file_name = "grayscale_" + directory + file_format
    grayscale_data = np.load(grayscale_file_name)

    return color_data, grayscale_data

def getData(directory, no_of_sunstones, no_of_not_sunstones, image_length, image_breadth):
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
    # stores 0 if the corresponding index in X does not has a sunstone image
    Y = np.append(np.ones(no_of_sunstones), np.zeros(no_of_not_sunstones))

    # Our data is extremely ordered right now: it has all the sunstones together followed by all not_sunstones
    # To make this training data less biased, let's simaltaneouly shuffle X and Y
    shuffler = np.random.permutation(m)
    X_Grayscale = X_Grayscale[shuffler, :]
    X_Color = X_Color[shuffler, :]
    Y = Y[shuffler]

    # Finally, we'll transpose Y to make it a column vector (to make life easier when dealing with matrix multiplications)
    Y = Y[..., None]

    return m, X_Grayscale, X_Color, Y


def accuracy(h, y):
    """Function to test the accuracy of a hypothesis.
       Input: h -> hypothesis
              y -> Actual/Real Values
       Output: Precision of h
               Amount of sunstones wasted
               Overall accuracy of the hypothesis """

    m = len(y)
    true_positives = len(y[y == 1])
    true_negatives = len(y[y == 0])

    hypothesis_positives = len(h[h >= 0.5])

    false_positives = 0
    false_negatives = 0

    for i in range(len(h)):
        if h[i] >= 0.5 and y[i] == 0:
            false_positives += 1
        elif h[i] < 0.5 and y[i] == 1:
            false_negatives += 1

    total_errors = false_positives + false_negatives

    precision = (hypothesis_positives - false_positives) / hypothesis_positives
    precision *= 100

    wastage = false_negatives * 100 / true_positives

    overall_accuracy = 100 - (total_errors * 100 / m)

    print("Precision of output =  " + str(precision) + "%")
    print("Sunstone Wastage =  " + str(wastage) + "%")
    print("Overall accuracy =  " + str(overall_accuracy) + "%", end = '\n\n')

    #return precision, wastage, overall_accuracy

def splitDataTraining(x, y):
    m = np.size(y)
    training_start = 0
    cv_start = int(0.6 * m)

    x_training = x[training_start:cv_start]
    y_training = y[training_start:cv_start]

    return x_training, y_training

def splitDataCV(x, y):
    m = np.size(y)
    cv_start = int(0.6 * m)
    test_start = int(0.8 * m)

    x_cv = x[cv_start:test_start]
    y_cv = y[cv_start:test_start]

    return x_cv, y_cv

def splitDataTesting(x, y):
    m = np.size(y)

    test_start = int(0.8 * m)

    x_test = x[test_start:]
    y_test = y[test_start:]

    return x_test, y_test

def LENET_accuracy(h, y):
    """Function to test the accuracy of a hypothesis.
       Input: h -> hypothesis
              y -> Actual/Real Values
       Output: Precision of h
               Amount of sunstones wasted
               Overall accuracy of the hypothesis """

    m = len(y)
    true_positives = len(y[y == 1])
    true_negatives = len(y[y == 0])

    hypothesis_positives = len(h[h >= 0.5])

    false_positives = 0
    false_negatives = 0

    for i in range(len(h)):
        if h[i,1] >= 0.5 and y[i] == 0:
            false_positives += 1
        elif h[i,1] < 0.5 and y[i] == 1:
            false_negatives += 1

    total_errors = false_positives + false_negatives

    precision = (hypothesis_positives - false_positives) / hypothesis_positives
    precision *= 100

    wastage = false_negatives * 100 / true_positives

    overall_accuracy = 100 - (total_errors * 100 / m)

    print("Precision of output =  " + str(precision) + "%")
    print("Sunstone Wastage =  " + str(wastage) + "%")
    print("Overall accuracy =  " + str(overall_accuracy) + "%", end = '\n\n')

    #return precision, wastage, overall_accuracy
