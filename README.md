# Lookout-CCTV-System

A CCTV system to detect humans entering restricted zones. Uses a Yolov11n model trained from COCO person dataset.

---

## Instalation
- Clone the repository
- Download the trained YOLO model in releases
- Place YOLO model in assets
- Install Dependencies in requirement.txt


---

## Usage Guide
- Install DroidCam on your phone
- start streaming, note WIFI IP and Port 
- Run the backend server main.py
- webpage automatically opens after
- Enter the WIFI IP:Port of the camera
- Connect to camera and start detecting

## Model Details

- **Model:** YOLOv11n  
- **Dataset:** COCO (only people class)  

---

## Limitations

- Detection can be slow on CPU for high-resolution video  
- YOLOv11n is lightweight which might not be accurate

---

## Future Improvements


