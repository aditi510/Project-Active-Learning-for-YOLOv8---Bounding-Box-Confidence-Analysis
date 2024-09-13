import os
from ultralytics import YOLO


model_path = 'D:\\Courses\\pythonProject\\pythonProjectCV\\yolov8s.pt'  

image_dir = 'D:\\Courses\\pythonProject\\pythonProjectCV\\Datasets\\Car_Pedestrian_detection.v1i.yolov8\\train\\images'


print("Loading model...")
model = YOLO(model_path)

confidence_threshold = 0.40


results = model.predict(source=image_dir, conf=confidence_threshold, save=True)  

print("Inference complete. Check the 'runs' directory for results.")