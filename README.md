# Bread Quality Grader using yolov3
## Vision Sensors and Perception 
#### By Sean Sminia and Ray Mindiola

### Introduction
This repository contains the project files for a vision sensors and perception assignment to identify bread within any picture and then give a grade depending on how edible the bread is. The possible assigned grades range from A (enjoyable) to F (un-edible).
To grant a grade to any given picture of bread, the following methology was applied to our project. 

The input picture undergoes object detection through the use a neural network (Yolo) to identify bread, in this case, specifically pistolets. This was achieved using the principles of transfer learning. Once the pistolets are identified, they are cropped from the image. 
Afterwards, they undergo detection again through the same neural network, with a smaller Intersection over Union (IOU) threshold. This causes multiple dections to happen over the same object. These detections are then exported and their confidence levels are used to build an estimate of quality given how cofident the network is that the bread is baked against the confidence of the bread being burnt or not baked at all.

The grade is then written as text on the original image before then being regenerated and rewritten to the project folder.

### Behaviour
The program beheaviour is described in the following flow chart.

![Flowchart](https://i.imgur.com/UVAJDfq.png)

**To give an illustration on how the program works, in this section, it will be taken step by tep.**

The input image:
![input image](https://i.imgur.com/HJxSM6B.jpg)

This image is then run through YOLO to identify the object
![Object found](https://i.imgur.com/M5lRe67.png)

Which is then cropped. A second run of the neural network is done with a lower IOU Threshold
![Second NN run](https://i.imgur.com/A2FHBKB.png)

Then, using these multiple detections, a simple algorithm is used to derivate a grade based on the results of the second Neural network. This Grade is then drawn over the origianl image
![Result](https://i.imgur.com/hqzfSoC.png)

### Requirements
Within this section you will find a list of the libraries requiered to run the scripts included in this project.

- cv2 ver 4.2.0
- numpy ver 1.17
- opencv-python ver 4.1 or higher
- torch ver 1.5 or higher
- torchvision
- matplotlib
- pycocotools
- tqdm
- pillow
- tensorboard ver 1.14 or higher

Note: The project runs on the ultralitycs version of yolov3, [available here](https://github.com/ultralytics/yolov3)
**The required libraries can be also found in "requierements.txt"**

### How to run
This section will briefly go over instructions to run the project.

**Crop.py is the main project file. Running this will create the graded bread pictures on the folder**

##### Requiered changes
some values within the script are hard coded with absolute paths, and as such, requiered to be modified to ensure the program runs as expected.

Under **main:**
- Make sure to modify the target file(s) path to fit your system.

Under the **detection class:**
- Make sure the configuration file (cfg), name file and weights file are properly assigned within your system

##### Executing the file:
after making the above mentioned changes. Using python, run the script called "Crop.py"

### Results
The script behaves as expected, and is capable of grading bread as intended. The image results can be found in the folder "model graded" given from this reposity. Additionally, the original input images can be found within the "model test" folder, also within this folder.

### Limitations
The project currently performs best when there is only 1 pistolete in the input picture. The quality of the second run of image detection contains noticeably lower confidence levels, this is due to training data set of the neural network included very few close up images of the pistoletes.  

### Recomendations
Idealy, it would be best to replace the second run of the object detection using yolo with a much simpler image proccesing method, given that once the image has been cropped, the noise that would normally obstruct the use of the simpler tools would be mostly eliminated. Alternatively another neural network dedicated to image classification could be used to further improve the results of the grading system. The current layout of Crop.py allows for the class the grading function is included to be altered to acomodate for these changes with relative ease. A new class can also be made to acomodate this new neural network. Unfortunately it was not possible to implement this in the project due to time constraints. An output folder for the images should be implemented, to avoid cluttering the main project folder with the graded bread pictures.
