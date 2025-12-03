from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
#from fastapi.templating import Jinja2Templates
import cv2
from ultralytics import YOLO
from threading import Thread
import asyncio
import threading, webbrowser
import time
import numpy as np
import uvicorn
import re




app = FastAPI()

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")
#templates = Jinja2Templates(directory="templates")

# Global variables
camera = None
camera_url = None
detecting = False
detected = False
output_frame = None
frame_count = 0
skip_frame = 4

model = YOLO("static/assets/HumanDetectorModel.pt")  # YOLO model

import threading
camera_lock = threading.Lock()

# ---------------------------
# Video capture thread
# ---------------------------
def capture_frames():
    global camera, output_frame, detecting, detected, frame_count
    default_image = cv2.imread("static/assets/images/no-image.jpg")
    default_image = cv2.resize(default_image, (640, 480))

    last_boxes = []
    color = 255
    frames_since_detected = 0

    while True:
        with camera_lock:
            if camera is not None and camera.isOpened():
                try:
                    ret, frame = camera.read()
                    if not ret:
                        output_frame = cv2.imencode(".jpg", default_image)[1].tobytes()
                        continue

                    # YOLO detection
                    if detecting:
                        frame_count += 1
                        
                        if frame_count%4 == 0:

                            orig_h, orig_w = frame.shape[:2]

                            resized = cv2.resize(frame, (640, 480))
                            results = model(resized)[0]

                            detected = len(results.boxes) > 0
                            if detected:
                                frames_since_detected = 0
                            else:
                                frames_since_detected += 1


                            scaled_x = orig_w / 640
                            scaled_y = orig_h / 480
                            scaled_boxes = []

                            for box in results.boxes.xyxy:
                                x1, y1, x2, y2 = box
                                scaled_boxes.append([
                                    x1*scaled_x, y1*scaled_y,
                                    x2*scaled_x, y2*scaled_y
                                ])
                            last_boxes = scaled_boxes
                            color = 255
                        else:
                            color = max(50, color-50)
                        
                        for box in last_boxes:
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, color), 2)
                            cv2.rectangle(frame, (x1, y1), (x1+100, y1-30), (0,0,0), -1)
                            cv2.putText(frame, "Person", (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                        

                            
                        if detected:
                            cv2.rectangle(frame, (10, 10), (250, 50), (0,0,170), -1)
                            cv2.putText(frame, "Person Detected", (20, 38),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                        
                        if frames_since_detected >= 30:
                            cv2.rectangle(frame, (10, 10), (300, 50), (0,170,0), -1)
                            cv2.putText(frame, "No People Detected", (20, 38),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                    output_frame = cv2.imencode(".jpg", frame, encode_param)[1].tobytes()
                
                except:
                    output_frame = cv2.imencode(".jpg", default_image)[1].tobytes()
                    camera.release()
                    camera = None    
            else:
                # No camera → show default image
                output_frame = cv2.imencode(".jpg", default_image)[1].tobytes()

        time.sleep(0.01)
thread = Thread(target=capture_frames)
thread.daemon = True
thread.start()

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")

def validate_camera_url(url: str) -> bool:
    pattern = r'^(http|rtsp)://\d{1,3}(\.\d{1,3}){3}(:\d+)?(/.*)?$'
    return re.match(pattern, url) is not None

# ---------------------------
# Routes
# ---------------------------
@app.get("/")
async def index(request: Request):
    #return templates.TemplateResponse("index.html", {"request": request})
    return FileResponse("index.html")

@app.get("/connect_camera")
async def connect_camera(url: str):
    global camera, camera_url
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
        
        if not validate_camera_url(url):
            return JSONResponse({"message": "Invalid camera URL!"})
        
        camera_url = url

        camera = cv2.VideoCapture(camera_url)
        if camera.isOpened():
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
            camera.set(cv2.CAP_PROP_FPS, 30)  
            return JSONResponse({"message": "Camera connected successfully!"})
        else:
            camera.release()
            camera = None
            return JSONResponse({"message": "Failed to open camera!"})

@app.get("/disconnect_camera")
async def disconnect_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
            return JSONResponse({"message": "Camera disconnected!"})
        else:
            return JSONResponse({"message": "No camera to disconnect."})


@app.get("/start_detection")
async def start_detection():
    global detecting
    if camera is None:
        return JSONResponse({"message": "No camera connected!"})
    else:
        detecting = True
        return JSONResponse({"message": "Detection started!"})

@app.get("/stop_detection")
async def stop_detection():
    global detecting
    global detected
    if camera is None:
        return JSONResponse({"message": "No camera connected!"})
    else:
        detecting = False
        detected = False
        return JSONResponse({"message": "Detection stopped!"})

@app.get("/detection_status")
async def detection_status():
    if detecting:
        return JSONResponse({"detected": detected})

@app.get("/video_feed")
async def video_feed():
    async def generate():
        global output_frame
        while True:
            if output_frame is None:
                await asyncio.sleep(0.033)
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
            await asyncio.sleep(0.033)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

LOG_FILE = "logs/logs.txt"  # write to this file

@app.post("/append_log")
async def append_log(data: dict):
    message = data.get("line", "")
    if message:
        # Append as new line (bottom)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    return ""

if __name__ == "__main__":
    
    threading.Timer(1, open_browser).start()
    
    uvicorn.run(
        "main:app",  # "module:app"
        host="0.0.0.0",
        port=5000,
    )
