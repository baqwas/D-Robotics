#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
helloobject.py
Use OpenCV and fcos model to detect an image from the attached webcam

This file is part of d-robotics repository at https://github.com/baqwas/d-robotics
It is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
This file is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with Foobar. If not,
see <https://www.gnu.org/licenses/>.

(C) 2024 ParkCircus Productions; All Rights Reserved

    Common FourCC codes:
        'XVID': A widely supported codec for AVI files
        'MJPG': Motion JPEG codec, often used for higher quality output
        'DIVX': Another popular codec for AVI files
        'MP4V': For MP4 files
        'H264': A modern and efficient codec, often used for high-quality video

    - Load the fcos object detection algorithm model (trained on the COCO dataset with 80 object categories).
    - Read the video stream from the USB camera and perform inference.
    - Parse the model output and render the results to the original video stream.
    - Output the rendered video stream via the HDMI interface.

The hobot_dnn model inference library is pre-installed on the Ubuntu system of the development board.
Users can import the module and check the version information.
$ sudo python3
Python 3.8.10 (default, Mar 15 2022, 12:22:08)
Type "help", "copyright", "credits" or "license" for more information.
>>> from hobot_dnn import pyeasy_dnn as dnn
>>> dir(dnn)
['Model', 'TensorProperties', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', 'load', 'pyDNNTensor']

The main classes and interfaces used in the hobot_dnn inference library are as follows:
    Model: AI algorithm model class, used for loading algorithm models and performing inference calculations. For more information, please refer to the Model documentation.
    pyDNNTensor: AI algorithm input and output data tensor class. For more information, please refer to the pyDNNTensor documentation.
    TensorProperties: Class for the properties of the input tensor of the model. For more information, please refer to the TensorProperties documentation.
    load: Load algorithm models. For more information, please refer to the API interface documentation.

Adapted from DF Robot RDK X3 documentation
@sa https://arxiv.org/pdf/1905.11946
@sa https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_convolutional-modern/googlenet.ipynb
"""
from hobot_dnn import pyeasy_dnn as dnn  # inference module
from hobot_vio import libsrcampy as srcampy  # video output module
import numpy as np
import cv2
import colorsys

from src.basic.test_efficientnasnet_m import print_properties

# load model files to return Model class
models = dnn.load('/app/pydev_demo/models/fcos_512x512_nv12.bin')

# output consists of 15 groups of data that represent the detected object bounding boxes
# input tensor properties
print_properties(models[0].inputs[0].properties)
# output tensor properties
for output in models[0].outputs:
    print_properties(output.properties)

"""
process data:
    Use OpenCV to open the USB camera device node /dev/video8, 
    get real-time images, and 
    resize the images to fit the input tensor size of the model.
    
    $ ls /dev/video*
/dev/video0  /dev/video2  /dev/video4  /dev/video6  /dev/video8
/dev/video1  /dev/video3  /dev/video5  /dev/video7
    $
"""
# open USB webcam at /dev/video8 - why 8?
webcam_index = 8
cap = cv2.VideoCapture(webcam_index)
if (not cap.isOpened()):
    print(f"Could not open USB webcam at {webcam_index}")
    exit(-1)

codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
cap.set(cv2.CAP_PROP_FOURCC, codec)
cap.set(cv2.CAP_PROP_FPS, 30)  # framew per second
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# OpenCV uses BGR format, convert correspondingly to NV12 format
nv12_data = bgr2nv12_opencv(resized_data)
"""
Model Inference
    Call the forward interface of the Model class for inference. 
    The model will output 15 sets of data representing the detected object bounding boxes.
"""
outputs = models[0].forward(nv12_data)
"""
Post-processing
    The post-processing function postprocess will process 
    the object category, bounding box, and confidence information output by the model.
"""
prediction_bbox = postprocess(outputs, input_shape, origin_img_shape=(1080, 1920))