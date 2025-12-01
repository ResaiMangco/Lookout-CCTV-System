from flask import Flask, Response, request, jsonify, render_template
import cv2
from threading import Thread
from ultralytics import YOLO

app = Flask(__name__)

# Global variables
camera = None
camera_url = None
detecting = False
model = YOLO("assets/best.pt")

output_frame = None

# ---------------------------
# Video capture thread
# ---------------------------
def capture_frames():
    global camera, output_frame, detecting
    while True:
        if camera is not None:
            ret, frame = camera.read()
            if not ret:
                continue

            # Run YOLO detection if enabled
            if detecting:
                results = model(frame)
                for r in results:
                    for box in r.boxes.xyxy:  # xyxy boxes
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
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
                return jsonify({"message": "Failed to open camera."})
        except Exception as e:
            return jsonify({"message": str(e)})
    else:
        return jsonify({"message": "Camera already connected."})

@app.route('/start_detection')
def start_detection():
    global detecting
    detecting = True
    return jsonify({"message": "Human detection started!"})

@app.route('/stop_detection')
def stop_detection():
    global detecting
    detecting = False
    return jsonify({"message": "Human detection stopped!"})

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
    app.run(host="0.0.0.0", port=5000, debug=True)
