from ultralytics import YOLO
model = YOLO("yolo11n.pt")
results = model("../images/tram_houston.jpg")