from ultralytics import YOLO
from os import remove
try:
    remove("datasets/dota8/labels/train.cache")
except:
    pass
model = YOLO("yolo11n-zaxis.yaml")
print(model.model.model)
print("loaded")
results = model.train(data="datasets/dota8/data.yaml", epochs=10, imgsz=640,)
print("test")