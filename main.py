from flask import Flask, Response, request, jsonify, render_template
import cv2
from threading import Thread
from sympy import false, true
from ultralytics import YOLO
import webbrowser
import threading


app = Flask(__name__)

# Global variables
camera = None
camera_url = None
detecting = False
detected = False
model = YOLO("assets/best.pt")
signal_counter = 0
signal_send = 4

output_frame = None

def open_browser():
    print("Opening Browser")
    webbrowser.open_new("http://127.0.0.1:5000")

# ---------------------------
# Video capture thread
# ---------------------------
def capture_frames():
    global camera, output_frame, detecting, detected, signal_counter
    while True:
        if camera is not None:
            ret, frame = camera.read()
            if not ret:
                continue

            # Run YOLO detection if enabled
            if detecting:                
                results = model(frame)[0]
                
                if len(results.boxes) > 0:
                    detected = True                        
                else:
                    detected = False
                    
                for box in results.boxes.xyxy:  # xyxy boxes
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Store frame
            ret, jpeg = cv2.imencode('.jpg', frame)
            output_frame = jpeg.tobytes()

# Start capture thread
thread = Thread(target=capture_frames)
thread.daemon = True
thread.start()

# ---------------------------
# Flask routes
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/connect_camera')
def connect_camera():
    global camera, camera_url
    camera_url = request.args.get('url')
    print(camera_url)
    if camera is None:
        try:
            camera = cv2.VideoCapture(camera_url)
            if camera.isOpened():
                return jsonify({"message": "Camera connected successfully!"})
            else:
                camera = None
                return jsonify({"message": "Failed to open camera!"})
        except Exception as e:
            return jsonify({"message": str(e)})
    else:
        return jsonify({"message": "Camera already connected!"})

@app.route('/start_detection')
def start_detection():
    global detecting, camera
    if camera is None:
        return jsonify({"message": "No Camera, detection cancelled!"})
    else:
        detecting = True
        return jsonify({"message": "Human detection started!"})

@app.route('/stop_detection')
def stop_detection():
    global detecting, camera
    if camera is None:
        return jsonify({"message": "No Camera, detection cancelled!"})
    else:
        detecting = False
        return jsonify({"message": "Human detection stopped!"})

@app.route('/disconnect_camera')
def disconnectCamera():
    global camera
    if not camera:
        return jsonify({"message": "No Camera connected!"})
    else:
        camera=None
        return jsonify({"message": "Camera Disconnected!"})
    
@app.route('/detection_status')
def detection_status():
    global detected
    return jsonify({detected:detected})

@app.route('/video_feed')
def video_feed():
    def generate():
        global output_frame
        while True:
            if output_frame is None:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Timer(1, open_browser).start()

    app.run(host="0.0.0.0", port=5000)
    

