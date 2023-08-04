import tensorflow as tf
import cv2
import numpy as np
import glob
import time
import Essentials

def set_input_tensor(interpretor, image):
    tensor_index = interpreter.get_input_details()[0]["index"]
    input_tensor = interpretor.tensor(tensor_index)()[0]
    input_tensor[:,:] = image

def clasify_image(interpretor, image):

    set_input_tensor(interpretor, image)

    interpretor.invoke()
    output_details = interpreter.get_output_details()[0]

    scores = interpreter.get_tensor(output_details['index'])[0]
    return np.max(np.unique(scores))

def inferAndGetData(directory, no_of_sunstones, no_of_not_sunstones, image_length, image_breadth):
    
    location = '/Users/evaanahmed/Desktop/Sunstone Sorter/RL040420'
    status = ['Sunstones', 'Not_Sunstones']
    normalization_Factor = 255.0
    i = 0
    totalTime = 0
    
    for val in status:
        for filename in glob.glob(location + '/' + val + '/*.jpg'):
            img_Color = cv2.imread(filename)
            start = time.time()
            img_Color = img_Color / normalization_Factor
            clasify_image(interpreter, img_Color)
            end = time.time()
            i = i+1
            totalTime += end-start
            
    return totalTime,i

print("Loading Model")

Interpreter = tf.lite.Interpreter
interpreter = Interpreter("model.tflite")
interpreter.allocate_tensors()


print("Loading & Inferencing Data One-By-One")
RL040420_Data = ["RL040420", 334, 3079, 112, 112]
totalTime, m = inferAndGetData(*RL040420_Data)
print("Process Completed For", m, "Images")

print("\nTime Elapsed: %.3f seconds\nTime Per Image: %.3f seconds" %
      (totalTime, totalTime/m))
