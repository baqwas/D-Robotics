#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_googlenet.py
Classify an image using the GoogLeNet model

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

Adapted from DF Robot RDK X3 documentation
@sa https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_convolutional-modern/googlenet.ipynb
"""
from ultralytics import YOLO
import argparse, glob, os

def list_image_files(folder):
    """
    Lists all image files within a given folder.

    Args:
      folder: The path to the directory containing the images.

    Returns:
      A list of strings, where each string is the full path to an image file.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]  # what extensions have I missed?
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder, f"*{ext}")))

    return image_files

def predict_folder(folder, yolo_model):
    """
    Use YOLO Predict mode to detect objects in images in a specified folder using a specified YOLO pretrained model

    :param folder: all images in this folder will be used for the detection task
    :param yolo_model:  pretrained YOLO model filename
    :return:
    """
                                            # Load a model
    print(f"Using model file {yolo_model}")
    model = YOLO(yolo_model)                # pretrained YOLO11n model

                                            # Run batched inference on a list of images
    all_images = list_image_files(folder)
    print(f"Collected all images in {folder}")
    results = model(all_images, stream=True)    # return a list of Results objects

                                            # Process results list
    for result in results:
        boxes = result.boxes                # Boxes object for bounding box outputs
        masks = result.masks                # Masks object for segmentation masks outputs
        keypoints = result.keypoints        # Keypoints object for pose outputs
        probs = result.probs                # Probs object for classification outputs
        obb = result.obb                    # Oriented boxes object for OBB outputs
        result.show()                       # display to screen
        result.save(filename="result.jpg")  # save to disk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test YOLO11 Predict mode with all image files in a folder")
    parser.add_argument("image_folder", type=str, nargs="?", help="Path to the folder of image files",
                        default="/home/sunrise/PycharmProjects/d-robotics/images")
    parser.add_argument("model_file", type=str, nargs="?", help="Model filename",
                        default="yolo11n.pt")
    args = parser.parse_args()
    image_folder = args.image_folder
    model_file = args.model_file
    predict_folder(image_folder, model_file)
