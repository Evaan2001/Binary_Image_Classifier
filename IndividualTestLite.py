"""
My final model is a TensorFlow Lite model which can't be invoked by the standard fit function.
To load a TensorFlow Lite (tfLite) model, we'll need the tflite.Interpreter class. This class has a 
load_model() method that takes the path to the TensorFlow Lite model file as its argument. Once the model 
is loaded, we can use the predict() method to make predictions.

The tfLite.predict() method takes an input tensor as its argument. The input tensor must have the same shape as the
input tensor that was used to train the model. The tfLite.predict() method returns an output tensor. The output 
tensor contains the predictions for the input tensor.

The code here applies my model one-by-one on the testing images and saves the model predictions. 
"""

# Import relevant packages
import tensorflow as tf
import cv2
import numpy as np
import glob
import time
import Essentials

def set_input_tensor(interpretor, image):
    """
    This function defines the dimensions for the input tensors
    
    Inputs:
        interpreter (tf.lite.Interpreter): A TensorFlow Lite interpreter object that has been initialized.
        image (numpy.ndarray): A 2-dimensional NumPy array representing the image to be set as the input tensor.

    Returns:
        None
    """
    tensor_index = interpreter.get_input_details()[0]["index"] # Retrieve the index of the input tensor from the interpreter object.
    input_tensor = interpretor.tensor(tensor_index)()[0]       # Obtain reference to the actual input tensor using 
    # The extra parentheses () after interpreter.tensor(tensor_index) is to get a callable that can be used to get a pointer to the input tensor data
    input_tensor[:,:] = image # copy the data from the image array into the input tensor, effectively setting the input tensor to the provided image data.

def classify_image(interpreter, image):
    """
    The function takes a TensorFlow Lite interpreter and an image as input and returns
    the maximum score from the output tensor after performing inference with the interpreter

    Inputs:
        interpreter (tf.lite.Interpreter) – A TensorFlow Lite interpreter object that has been initialized
        image (numpy.ndarray) – A 2-dimensional NumPy array representing the image to be classified

    Returns:
        max_score – max score from the output tensor after performing inference with the interpreter
    """

    # Set the input tensor of the interpreter to the given image
    set_input_tensor(interpreter, image)

    # Perform inference using the interpreter
    interpreter.invoke()

    # Retrieve the output details of the interpreter
    output_details = interpreter.get_output_details()[0]

    # Get the scores from the output tensor of the interpreter
    scores = interpreter.get_tensor(output_details['index'])[0]

    # Extract the maximum score from the unique scores
    max_score = np.max(np.unique(scores))

    # Return the max score
    return max_score

def inferAndGetData(directory, no_of_sunstones, no_of_not_sunstones, image_length, image_breadth):
    """
    This function retrieves all of the images from a directory and runs the TensorFlow Lite model on them one-by-one
    Inputs:
        directory – path to the directory containing the images
        no_of_sunstones – number of images that have a sunstone
        no_of_not_sunstones – number of images that don't have a sunstone
        image_length – the width of an image
        image_breadth – the height of an image
    Outputs:
        totalTime – time taken to run the model on all of the images
        i – no of images processed
    """
    
    location = '/Users/evaanahmed/Desktop/Sunstone Sorter/RL040420' # Just for my local device :)
    status = ['Sunstones', 'Not_Sunstones']
    normalization_Factor = 255.0 # To limit the pixel brightness values between 0 and 1
    i = 0         # To store how many images have been processed
    totalTime = 0 # Count how long it took the model to run for all of the images

    # Loop through all of the relevant directories
    for val in status: 
        for filename in glob.glob(location + '/' + val + '/*.jpg'):
            img_Color = cv2.imread(filename)             # Retrieve an image
            start = time.time()                          # Start the timer
            img_Color = img_Color / normalization_Factor # Normalize the image
            clasify_image(interpreter, img_Color)        # Run the model
            end = time.time()                            # Stop the timer
            i = i+1                                      # Increment the image count
            totalTime += end-start                       # Add the time taken for this image to our total time calculation

    # Return outputs
    return totalTime, i

print("Loading Model")

Interpreter = tf.lite.Interpreter         # Load the interpreter 
interpreter = Interpreter("model.tflite") # Access the tf Lite model
interpreter.allocate_tensors()            # pre-plan tensor allocations to optimize inferencing

print("Loading & Inferencing Data One-By-One")

# Set up parameters to call our infer function
DIRECTORY_NAME = "RL040420"
num_positive_cases = 334 
num_negative_cases = 3079
IMAGE_LENGTH = 112
IMAGE_BREADTH = 112
DIRECTORY_DATA = [DIRECTORY_NAME, num_positive_cases, num_negative_cases, IMAGE_LENGTH, IMAGE_BREADTH]
totalTime, m = inferAndGetData(*DIRECTORY_DATA)

# Display final results
print("Process Completed For", m, "Images")
print("\nTime Elapsed: %.3f seconds\nTime Per Image: %.3f seconds" %
      (totalTime, totalTime/m))
