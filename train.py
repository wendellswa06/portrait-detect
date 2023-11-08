import torch
from ultralytics import YOLO

model = YOLO('yolov8l.pt')


if __name__ == "__main__":
    results = model.train(data='./data.yaml', name='Bottles', epochs=20, batch=16, cache=True, imgsz=640, iou=0.5, augment=True, degrees=25.0, fliplr=0.0, lr0=0.0001, optimizer='Adam')
