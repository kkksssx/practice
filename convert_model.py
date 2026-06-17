from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(format = "openvino", half = False)
print("модель конвертирована в openvino")