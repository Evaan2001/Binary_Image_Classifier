
<p  align="center">
<img  src= "https://github.com/Evaan2001/Binary_Image_Classifier/assets/82547698/510a60c1-55eb-485e-8064-ef7515144dac"
width = "900"/>
  
<h1 align="center">
Binary Image Classifier
</h1>

<h3 align="center">
I developed this classifier for a mining company to identify sunstones (a gemstone used for jewellery) from a mix of other rocks
</h3>

<h2 align="center"> 
Preliminary Context</h2>

<p align="center">
Sunstones are rocks that glow when a laser light passes through them. The first step in the mining process is to seperate out the rocks that glew under laser from those that didn't – so to sort out the sunstone rocks. The mining company I was working for wanted to automate this sorting process and had developed a prototype (do check out the video in this repository – "2nd Summer.mov" – to see the prototype work). They had a conveyer belt with a laser light and camera in the center and air pistons on one end; everything was connected by a Raspberry PI. The idea was that rocks will be laid out on the conveyer belt, the laser light will always be on, the camera will continuously take pictures, and if we take pictures that have glowing rocks, we will trigger the air pistons to filter them out. 
</p>

<p  align="center">
My job was to come up with a binary image classifier that would run on the Raspberry PI and identify sunstones (aka, glowing rocks) in real time.
</p>

<h2 align="center"> 
Training Data/Images
</h2>

<p  align="center">
I knew a CNN would be well suited for the task. To set up the training data, we saved a lot of pictures taken by our camera. We had 334 images that had a sunstone present. And we had 3079 images that had no sunstone present. Here are 2 sample images:
</p>

<p  align="center">
<img  src= "https://github.com/Evaan2001/Binary_Image_Classifier/assets/82547698/ec6d25d5-bba8-431f-bc30-f829730524b5"
  width="500">
</p>

<h2  align="center">
Model Structure
</h2>

<p  align="center">
T
</p>

<h2  align="center">
Model Training
</h2>
