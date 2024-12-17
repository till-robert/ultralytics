from ultralytics import YOLO
# from os import remove
# try:
#     remove("data_gen/Dataset/labels/train.cache")
#     remove("data_gen/Dataset/labels/val.cache")
# except:
#     pass
model = YOLO("runs/zaxis/train106/weights/best.pt")
# print(model.model.model)
print("loaded")
# results = model.train(data="data_gen/data.yaml", epochs=100, imgsz=512,)
results = model.val()

print("test")