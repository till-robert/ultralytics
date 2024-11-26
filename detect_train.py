from ultralytics import YOLO
from os import remove
try:
    remove("data_gen/Dataset/labels/train.cache")
    remove("data_gen/Dataset/labels/val.cache")
except:
    pass
model = YOLO("yolo11n.yaml")
print(model.model.model)
print("loaded")
results = model.train(data="data_gen/data.yaml", epochs=42, imgsz=512,)
print("test")