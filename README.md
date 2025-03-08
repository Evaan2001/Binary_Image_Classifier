<p  align="center">
<img  src= "https://github.com/Evaan2001/Images_For_ReadMe/blob/main/Binary_Classifier.png"
width = "900"/>
  
<h1 align="center">
Binary Image Classifier
</h1>

<h3 align="center">
I developed this classifier for a mining company to identify sunstones (a gemstone used for jewelry) from a mix of other rocks
</h3>

<h2 align="center"> 
Preliminary Context</h2>

<p align="center">
Sunstones are rocks that glow when a laser light passes through them. The first step in the mining process is to separate out the rocks that glew under laser from those that didn't – so to sort out the sunstone rocks. The mining company I was working for wanted to automate this sorting process and had developed a prototype (do check out the video in this repository – "2nd Summer.mov" – to see the prototype work). They had a conveyer belt with a laser light and camera in the center and air pistons on one end; everything was connected by a Raspberry PI. The idea was that rocks will be laid out on the conveyer belt, the laser light will always be on, the camera will continuously take pictures, and if we take pictures that have glowing rocks, we will trigger the air pistons to filter them out. 
</p>

<p  align="center">
My job was to come up with a binary image classifier that would run on the Raspberry PI and identify sunstones (aka, glowing rocks) in real-time.
</p>

<h2 align="center"> 
Working Video Of The Prototype
</h2>

https://github.com/user-attachments/assets/809ec6c6-a88f-4e32-b1db-0fae9541b183

<h2 align="center"> 
Training Data/Images
</h2>

<p  align="center">
I knew a CNN would be well suited for the task. To set up the training data, we saved a lot of pictures taken by our camera. We had 334 images that had a sunstone present. And we had 3079 images that had no sunstone present. Here are 2 sample images:
</p>

<p  align="center">
<img  src= "https://github.com/Evaan2001/Images_For_ReadMe/blob/main/Binary_Classifier_Training_Photo.png"
  width="500">
</p>

<h2  align="center">
Model Structure
</h2>

<p  align="center">
I knew that my model would be running on a Raspberry PI. This meant my model won't have access to abundant computational resources and thus needed to be lightweight. I used TensorFlow (in Python) to set up the model architecture. And here are the different layers –
</p>

1. 1st convolutional layer with 32 3*3 kernels
2. 1st activation layer using relu
3. 1st MaxPooling layer 
4. 2nd convolutional layer with 32 3*3 kernels again
5. 2nd activation layer using relu again
6. 2nd MaxPooling layer 
7. Model Flattening
8. Dense (aka, fully connected) layer with 64 nodes
9. 3rd activation layer using relu again
10. Dropout layer
11. Dense layer with 1 node (to calculate the weighted sum of previous layer nodes)
12. 4th Activation layer, this time using sigmoid 

<h2  align="center">
Notes On Model Training
</h2>

1. **Data Format** – Loading images every time to train multiple models was very time-consuming. So I chose to first load the images into numpy arrays and then store
them in a single compressed "npz" file. This is a compressed file format 
that allows us to save and load multiple arrays efficiently. Unfortunately, GitHub won't let me upload the npz file containing the original color images. So I provided the npz file containing grayscale images. But my model does need color images; sigh!
2. **Unbalanced Data** – Our data is heavily unbalanced. We have only 334 samples of sunstone images but 3074 not_sunstone images (that's a 10x difference!). If we train our model with the data set, it will hardly learn how to identify sunstones as all the information it's getting is primarily for not_sunstone. To solve this, we will set class weights. We want the model to get an equal amount of info on sunstones & not_sunstones to prevent overfitting. For each sunstone image, we have 3074/334 ≈ 9.2 images. So we'll instruct the model to consider each sunstone image as approximately 9.2 non_sunstone images. All we need to do is set class weights as  weights = [1, 9.2]. Here, the weight for class 0 (aka, not_sunstone) is 1 and the weight for class 1 (aka, sunstone) is 9.2
3. **Training Checkpoints** – We will train the model over 20 epochs. In case later epochs cause overfitting, we will save the weights after each epoch as checkpoints. This will help us retrieve the best-performing model when training is over.

<h2 align="center"> 
TensorFlow Lite
</h2>
 
<p  align="center">
I had to run my CNN on a Raspberry Pi 2 without a stable internet connection. To make sure my program does not freeze, I decided to convert my TensorFlow model into a TensorFlow Lite model. TensorFlow Lite is a lightweight version of TensorFlow that is designed for mobile and embedded devices. It provides a set of tools that enable on-device machine learning.
</p>

<p  align="center">
Using TensorFlow Lite models is not so straightforward. Firstly, to load a TensorFlow Lite (tfLite) model, we'll need the tflite.Interpreter class. This class has a load_model() method that takes the path to the TensorFlow Lite model file as its argument. Once the model is loaded, we can use the predict() method to make predictions. The tfLite.predict() method takes an input tensor as its argument. The input tensor must have the same shape as the input tensor that was used to train the model. The tfLite.predict() method returns an output tensor. The output tensor contains the predictions for the input tensor.
</p>

<h2 align="center"> 
Files
</h2>
 
<p  align="center">
Here's what you'll find –
</p>

1. *2nd Summer.mov* – A working video of the prototype. Sadly, GitHub won't play the video in your browser. So you'll have to download it and then open it. 😅 
2. *Binary_Image_Classifier.py* – This is the file that builds and trains the CNN. It also plots stats about the model's loss, accuracy, and false-negative rate across the training epochs
3. *Converter.py* – This file loads a TensorFlow model and converts it into a TensorFlow Lite model
4. *CreateData.py* – The file that loads the training images, both in color and grayscale, and stores them as compressed numpy arrays 
5. *Essentials.py* – A file that has a bunch of helper functions I used frequently
6. *IndividualTestLite.py* – The file that invokes the TensorFlow Lite model and uses it to predict the class of all the images
7. *grayscale_RL040420.npz* – The npz file containing the grayscale images
