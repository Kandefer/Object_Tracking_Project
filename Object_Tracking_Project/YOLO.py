# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:17:31 2022

@author: kxkfg2
"""


import cv2
import matplotlib.pyplot as plt

from utils import *
from darknet import Darknet


#%%
# Set the location and name of the cfg file
cfg_file = './cfg/yolov3.cfg'

# Set the location and name of the pre-trained weights file
weight_file = './weights/yolov3.weights'

# Set the location and name of the COCO object classes file
namesfile = 'data/coco.names'

# Load the network architecture
m = Darknet(cfg_file)

# Load the pre-trained weights
m.load_weights(weight_file)

# Load the COCO object classes
class_names = load_class_names(namesfile)

#%%
# Taking a Look at The Neural Network


# Print the neural network used in YOLOv3
m.print_network()

#%%
# As we can see, the neural network used by YOLOv3 consists mainly of convolutional layers, with some shortcut connections and upsample layers. For a full description of this network please refer to the <a href="https://pjreddie.com/media/files/papers/YOLOv3.pdf">YOLOv3 Paper</a>.

# Loading and Resizing Our Images

# In the code below, we load our images using OpenCV's `cv2.imread()` function. Since, this function loads images as BGR we will convert our images to RGB so we can display them with the correct colors.

# As we can see in the previous cell, the input size of the first layer of the network is 416 x 416 x 3. Since images have different sizes, we have to resize our images to be compatible with the input size of the first layer in the network. In the code below, we resize our images using OpenCV's `cv2.resize()` function. We then plot the original and resized images. 

# Set the default figure size
plt.rcParams['figure.figsize'] = [24.0, 14.0]

# Load the image
# img = cv2.imread('./images/dog.jpg')
# img = cv2.imread('./images/city_scene.jpg')
# img = cv2.imread('./images/korwin.jpg')
# img = cv2.imread('./images/korwin2.jpg')
# img = cv2.imread('./images/gru.jpg')
img = 'track_seq/track00100.jpg'


# Convert the image to RGB
original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# We resize the image to the input width and height of the first layer of the network.    
resized_image = cv2.resize(original_image, (m.width, m.height))

# Display the images
plt.subplot(121)
plt.title('Original Image')
plt.imshow(original_image)
plt.subplot(122)
plt.title('Resized Image')
plt.imshow(resized_image)
plt.show()

#%%
# Setting the Non-Maximal Suppression Threshold

# As you learned in the previous lessons, YOLO uses **Non-Maximal Suppression (NMS)** to only keep the best bounding box. The first step in NMS is to remove all the predicted bounding boxes that have a detection probability that is less than a given NMS threshold.  In the code below, we set this NMS threshold to `0.6`. This means that all predicted bounding boxes that have a detection probability less than 0.6 will be removed. 

# Set the NMS threshold
nms_thresh = 0.6  
#%%
# Setting the Intersection Over Union Threshold

# After removing all the predicted bounding boxes that have a low detection probability, the second step in NMS, is to select the bounding boxes with the highest detection probability and eliminate all the bounding boxes whose **Intersection Over Union (IOU)** value is higher than a given IOU threshold. In the code below, we set this IOU threshold to `0.4`. This means that all predicted bounding boxes that have an IOU value greater than 0.4 with respect to the best bounding boxes will be removed.

# In the `utils` module you will find the `nms` function, that performs the second step of Non-Maximal Suppression, and the `boxes_iou` function that calculates the Intersection over Union of two given bounding boxes. You are encouraged to look at these functions to see how they work. 

# Set the IOU threshold
iou_thresh = 0.4
#%%
# Object Detection

# Set the default figure size
plt.rcParams['figure.figsize'] = [24.0, 14.0]

for i in range(101,201):
    # Load the image
    # img = cv2.imread('./images/dog.jpg')
    # img = cv2.imread('./images/city_scene.jpg')
    # img = cv2.imread('./images/korwin.jpg')
    # img = cv2.imread('./images/korwin2.jpg')
    # img = cv2.imread('./images/gru.jpg')
    img = 'track_seq/track00' + str(i) + '.jpg'

    # Convert the image to RGB
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # We resize the image to the input width and height of the first layer of the network.    
    resized_image = cv2.resize(original_image, (m.width, m.height))
    
    # Set the IOU threshold. Default value is 0.4
    iou_thresh = 0.4
    
    # Set the NMS threshold. Default value is 0.6
    nms_thresh = 0.6
    
    # Detect objects in the image
    boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
    
    # Print the objects found and the confidence level
    print_objects(boxes, class_names)
    
    #Plot the image with bounding boxes and corresponding object class labels
    plot_boxes(original_image, boxes, class_names, plot_labels = True)
#%%

# =============================================================================
# camera = cv2.VideoCapture('./videos/test.mp4')
# 
# while True:
#     _,img = camera.read()
#     # Convert the image to RGB
#     original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 
#     # We resize the image to the input width and height of the first layer of the network.    
#     resized_image = cv2.resize(original_image, (m.width, m.height))
# 
#     # Set the IOU threshold. Default value is 0.4
#     iou_thresh = 0.4
# 
#     # Set the NMS threshold. Default value is 0.6
#     nms_thresh = 0.6
# 
#     # Detect objects in the image
#     boxes = detect_objects(m, resized_image, iou_thresh, nms_thresh)
# 
#     # Print the objects found and the confidence level
#     print_objects(boxes, class_names)
# 
#     #Plot the image with bounding boxes and corresponding object class labels
#     plot_boxes(original_image, boxes, class_names, plot_labels = True)
#     cv2.imshow("Image", img)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# =============================================================================
