from ultralytics import YOLO

model_path = r'D:\Courses\pythonProject\pythonProjectCV\Object_Detection\yolov8s.pt'
model = YOLO(model_path)

image_path = r'D:\Courses\pythonProject\pythonProjectCV\Object_Detection\ids_image_57312_2024-02-27_13-04-16.bmp'

results = model(image_path)

# The results for the first image in the results list
results[0].show()

#  the results to the output directory
output_directory = r'D:\Courses\pythonProject\pythonProjectCV\Object_Detection\output'
results[0].save(output_directory)
