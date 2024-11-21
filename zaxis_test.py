from ultralytics import YOLO
from os import remove
try:
    remove("datasets/dota8/labels/train.cache")
    remove("datasets/dota8/labels/val.cache")
except:
    pass
model = YOLO("yolo11n-zaxis.yaml")
print(model.model.model)
print("loaded")
results = model.train(data="data_gen/data.yaml", epochs=100, imgsz=512,)
print("test")
