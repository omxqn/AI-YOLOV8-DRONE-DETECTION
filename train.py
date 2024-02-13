# pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install ultralytics
import torch
import torchvision

from ultralytics import YOLO

model = YOLO("yolov8m.pt")
#result = model.train(data="data.yaml", epochs=100, imgz= 640, device=0,workers=0)


# to check


results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image






