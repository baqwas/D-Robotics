#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_mobilenet_unet.py
Use MobileNet for segmentation test

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

Efficient-NASNet-M focuses on efficiency and accuracy through:
- Efficient-NASNet-M leverages EfficientNet scaling moethod. (depth, width & resolution)
- NASNet architecture
- Improved resource usage
- Better performance without compromising accuracy

Adapted from DF Robot RDK X3 documentation
@sa https://arxiv.org/pdf/1905.11946
@sa https://colab.research.google.com/github/d2l-ai/d2l-pytorch-colab/blob/master/chapter_convolutional-modern/googlenet.ipynb
"""

import argparse
from hobot_dnn import pyeasy_dnn
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt

def bgr2nv12_opencv(image):
    height, width = image.shape[0], image.shape[1]
    area = height * width
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
    y = yuv420p[:area]
    uv_planar = yuv420p[area:].reshape((2, area // 4))
    uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))

    nv12 = np.zeros_like(yuv420p)
    nv12[:height * width] = y
    nv12[height * width:] = uv_packed
    return nv12

def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]


def plot_image(origin_image, onnx_output):
    def get_pallete():
        pallete = [
            128,
            64,
            128,
            244,
            35,
            232,
            70,
            70,
            70,
            102,
            102,
            156,
            190,
            153,
            153,
            153,
            153,
            153,
            250,
            170,
            30,
            220,
            220,
            0,
            107,
            142,
            35,
            152,
            251,
            152,
            0,
            130,
            180,
            220,
            20,
            60,
            255,
            0,
            0,
            0,
            0,
            142,
            0,
            0,
            70,
            0,
            60,
            100,
            0,
            80,
            100,
            0,
            0,
            230,
            119,
            11,
            32,
        ]
        return pallete

    onnx_output = onnx_output.astype(np.uint8)
    onnx_output = np.squeeze(onnx_output)

    image_shape = origin_image.shape[:2][::-1]

    onnx_output = np.expand_dims(onnx_output, axis=2)
    onnx_output = cv2.resize(onnx_output,
                             image_shape,
                             interpolation=cv2.INTER_NEAREST)
    out_img = Image.fromarray(onnx_output)
    out_img.putpalette(get_pallete())
    plt.imshow(origin_image)
    plt.imshow(out_img, alpha=0.6)
    fig_name = 'segment_result.png'
    print(f"Saving predicted image with name {fig_name} ")
    plt.savefig(fig_name)


def postprocess(model_output, origin_image):
    pred_result = np.argmax(model_output[0], axis=-1)
    print("=" * 10, "Postprocess successfully.", "=" * 10)
    print("=" * 10, "Waiting for drawing image ", "." * 10)
    plot_image(origin_image, pred_result)
    print("=" * 10, "Dump result image segment_result.png successfully.", "=" * 10)

if __name__ == '__main__':
    print(f"test_mobilenet_unet: v0.1")
    parser = argparse.ArgumentParser(description="Test MobileNet Unet segmentation")
    parser.add_argument("image_file", type=str, help="Path to the image file to be classified",
                        default="/home/sunrise/PycharmProjects/d-robotics/src/segmentation/segmentation.png")
    parser.add_argument("model_file", type=str, help="Path to the model file, i.e. EfficientNet-m",
                        default="/app/pydev_demo/models/mobilenet_unet_1024x2048_nv12.bin")
    args = parser.parse_args()
    image_file = args.image_file
    model_file = args.model_file
    # test classification result
    models = pyeasy_dnn.load(model_file)
    print("=" * 10, "Model load successfully.", "=" * 10)
    h, w = get_hw(models[0].inputs[0].properties)
    img_file = cv2.imread(image_file)
    des_dim = (w, h)
    resized_data = cv2.resize(img_file, des_dim, interpolation=cv2.INTER_AREA)
    nv12_data = bgr2nv12_opencv(resized_data)
    outputs = models[0].forward(nv12_data)
    print("=" * 10, "Model forward finished.", "=" * 10)
    postprocess(outputs[0].buffer, img_file)
