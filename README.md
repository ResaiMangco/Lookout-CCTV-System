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
Architecture
- Model: YOLOv11n
- Layers: 100
- Parameters: 2,582,347

Training Process
- Training Images: 12,823 (20% of COCO person class training set)
- Validation Images: 2,693
- Epochs: 60
- Batch Size: 16
- Image Size: 640x640
- Optimizer: AdamW
- Loss Functions: Boxes, Classification, DFL

Evalutation metrics
- Precision: 0.777
- Recall: 0.625
- mAP@50: 0719
- mAP@50-95: 0.474

---

## Limitations

- Detection can be slow on CPU for high-resolution video  
- YOLOv11n is lightweight which might not be accurate

---

## Future Improvements
- Addition of more Cameras
- ID system for multiple people detected with commands for IDs
- Synchronous camera detection intersection

