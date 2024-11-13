from ultralytics import YOLO
from os import remove
try:
    remove("datasets/dota8/labels/train.cache")
    remove("datasets/dota8/labels/val.cache")
except:
    pass
model = YOLO("yolo11n-zaxis.yaml")
results = model.train(data="datasets/dota8/data.yaml", epochs=100, imgsz=640,)