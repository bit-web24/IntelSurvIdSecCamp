import cv2
import base64
import json
import threading
import time
from flask import Flask, render_template, Response, request, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
import numpy as np
from ultralytics import YOLO
from dominant_color import detect_objects_on_image
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load your model
model = YOLO("models/best-cloth.pt")

class CameraHandler:
    def __init__(self):
        self.cap = None
        self.is_active = False
        self.latest_frame = None
        self.latest_detections = []
        self.lock = threading.Lock()
        
    def start_camera(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_active = True
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        with self.lock:
            self.is_active = False
            if self.cap:
                self.cap.release()
                self.cap = None
    
    def get_frame_with_detections(self):
        if not self.cap or not self.is_active:
            return None, []
        
        try:
            success, frame = self.cap.read()
            if not success:
                return None, []
            
            # Convert BGR frame to RGB PIL Image for detection
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Detect objects
            boxes = detect_objects_on_image(model, pil_img)
            
            # Draw detections on frame
            detected_objects = []
            for box in boxes:
                x1, y1, x2, y2, label, prob = box[:6]
                color_hex = box[6] if len(box) > 6 else None
                department = box[7] if len(box) > 7 else None
                
                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Prepare label text
                label_text = f"{label}"
                if department and department != "Unknown":
                    label_text = f"{department}"
                
                # Draw label with background for better visibility
                text_size = cv2.getTextSize(f"{label_text} {prob:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), (0, 255, 0), -1)
                cv2.putText(frame, f"{label_text} {prob:.2f}", (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Store detection info
                detected_objects.append({
                    'label': label,
                    'department': department,
                    'confidence': prob,
                    'color': color_hex,
                    'bbox': [x1, y1, x2, y2]
                })
            
            return frame, detected_objects
        except Exception as e:
            print(f"Error in get_frame_with_detections: {e}")
            return None, []

# Global camera handler
camera_handler = CameraHandler()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_handler
    if camera_handler.start_camera():
        return jsonify({'status': 'success', 'message': 'Camera started'})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to start camera'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_handler
    camera_handler.stop_camera()
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

def generate_frames():
    global camera_handler
    while True:
        if camera_handler.is_active:
            frame, detections = camera_handler.get_frame_with_detections()
            if frame is not None:
                try:
                    # Convert frame to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    print(f"Error encoding frame: {e}")
        else:
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def detection_worker():
    """Background thread to send detection data to clients"""
    while True:
        try:
            if camera_handler.is_active:
                frame, detections = camera_handler.get_frame_with_detections()
                if detections:
                    socketio.emit('detections_update', {
                        'detections': detections,
                        'timestamp': time.time()
                    })
            time.sleep(0.5)  # Update every 500ms
        except Exception as e:
            print(f"Error in detection worker: {e}")
            time.sleep(1)

# Start background detection thread
detection_thread = threading.Thread(target=detection_worker, daemon=True)
detection_thread.start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
