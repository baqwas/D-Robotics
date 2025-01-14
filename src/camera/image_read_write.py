#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_read_write.py
Use OpenCV to read and write  an image

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

Adapted from DF Robot RDK X3 documentation
@sa https://arxiv.org/pdf/1905.11946
@sa https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_convolutional-modern/googlenet.ipynb
"""

import argparse
import numpy as np
import cv2

def image_read_write(file_name):
    # 打开 usb camera: /dev/video8
    cap = cv2.VideoCapture(8)
    if cap.isOpened():
        print("USB camera opened")
    else:
        exit(-1)

    codec = cv2.VideoWriter_fourcc( 'M', 'J', 'P', 'G' )
    cap.set(cv2.CAP_PROP_FOURCC, codec)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    _ ,frame = cap.read()

    if frame is not None:
        cv2.imwrite(file_name, frame)
    else:
        print("Failed to retrieve frame from USB camera")

    return

if __name__ == '__main__':
    print(f"image_read_write: v0.1")
    parser = argparse.ArgumentParser(description="Test EfficientNet-m model classification")
    parser.add_argument("image_file", type=str, help="Path to the image file to be classified",
                        default="/home/sunrise/PycharmProjects/d-robotics/src/camera/image_read_write.jpg")
    args = parser.parse_args()
    image_file = args.image_file
    print(f"Current frame from camera to be written as {image_file} in current folder")

    image_read_write(image_file)