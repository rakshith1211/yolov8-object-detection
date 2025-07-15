import torch
from pathlib import Path
import cv2
import shutil
from matplotlib import pyplot as plt
from ultralytics import YOLO

# Load YOLOv8m model
model = YOLO("yolov8m.pt")

# Path to input image
image_path = "1.jpeg"

# Run detection
results = model(image_path)

# Save results (YOLO auto-saves to runs/detect/exp/)
results[0].save()

# Get path to YOLO saved image with bounding boxes
saved_dir = results[0].save_dir  # This gives the full path to runs/detect/exp/
saved_image_path =Path(image_path).name

# Desired output path in the same directory
output_path = Path("op.jpg")

# Copy the detected image to "op.jpg"
shutil.copy(str(saved_image_path), str(output_path))

# Read and display the saved image
img_with_boxes = cv2.imread(str(output_path))
plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
plt.title('Detected Objects')
plt.axis('off')
plt.show()
