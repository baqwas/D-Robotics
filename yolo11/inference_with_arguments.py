"""

@sa https://docs.ultralytics.com/modes/predict/#inference-sources
"""
from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("../images/tram_houston.jpg", save=True, imgsz=320, conf=0.5)