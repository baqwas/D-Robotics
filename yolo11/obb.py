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

# Load a pretrained YOLO11n model
model = YOLO("runs/yolo11n-obb.pt")

# Run inference on an image
results = model("../images/tram_houston.jpg")  # results list

# View results
for r in results:
    print(r.obb)  # print the OBB object containing the oriented detection bounding boxes