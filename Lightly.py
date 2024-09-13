import numpy as np
import torch
from ultralytics import YOLO
import sys
import cv2
import matplotlib.pyplot as plt
print(f"NumPy version: {np.__version__}")
print(f"NumPy ndarray: {np.ndarray}")
print(f"PyTorch version: {torch.__version__}")
print(f"Python executable: {sys.executable}")

# Path configurations
model_path = 'yolov8s.pt'
image_path = 'D:\Courses\pythonProject\pythonProjectCV\ids_image_57312_2024-02-27_13-04-16.bmp'


# Initialize YOLOv8 model
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"Error initializing YOLO model: {e}")
    sys.exit(1)

# object detection
try:
    results = model(image_path)
except Exception as e:
    print(f"Error performing detection: {e}")
    sys.exit(1)

image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

for result in results:
    for box in result.boxes:

      x_min, y_min, x_max, y_max = map(int, box.xyxy[0])  
      label = result.names[int(box.cls)] 
      confidence = box.conf[0].item() 


    cv2.rectangle(image_rgb, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    label_text = f"{label} {confidence:.2f}"
    
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(image_rgb, (x_min, y_min - 20), (x_min + w, y_min), (255, 0, 0), -1)
    cv2.putText(image_rgb, label_text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
plt.show()

