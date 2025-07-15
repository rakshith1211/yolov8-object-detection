# 🔍 Object Detection using YOLOv8 and Flask

This project demonstrates **real-time object detection** using **YOLOv8** deep learning models integrated with a **Flask web application**. It captures live webcam footage or uses static images to detect and label objects with high accuracy. The system is lightweight, easy to deploy, and ideal for smart surveillance, automation, and AI-powered monitoring.

---

## 🚀 Features

- Real-time object detection from webcam
- Detection on static images
- Flask-based web interface
- Uses pre-trained YOLOv8 models (`yolov8n.pt`, `yolov8m.pt`, `yolov8x.pt`)
- Easy to customize and extend

---

## 🖥️ Demo Preview

![Detection Image](static/1.jpeg)

---

## 📁 Project Structure

Vehicle-Detection-YOLOv8/
│
├── app.py # Flask web app for live webcam detection
├── model.py # Script for image-based object detection
├── templates/
│ └── index.html # Web UI for the live feed
├── static/
│ ├── 1.jpeg # Sample input image
│ ├── 2.jpeg # Another sample image
├── yolov8n.pt # YOLOv8 nano model
├── yolov8m.pt # YOLOv8 medium model
├── yolov8x.pt # YOLOv8 extra-large model
├── requirements.txt # Python dependencies
└── README.md # Project documentation
