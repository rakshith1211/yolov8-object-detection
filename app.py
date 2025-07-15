from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import time

app = Flask(__name__)
model = YOLO("yolov8n.pt")  # Lightweight model for speed
camera = cv2.VideoCapture(0)

def generate_frames():
    prev_time = 0
    frame_skip = 1
    count = 0

    while True:
        success, frame = camera.read()
        if not success:
            break

        resized_frame = cv2.resize(frame, (640, 480))

        current_time = time.time()
        if (current_time - prev_time) < 0.06:
            continue
        prev_time = current_time

        count += 1
        if count % frame_skip != 0:
            continue

        results = model.predict(source=resized_frame, conf=0.5, verbose=False)
        annotated = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
