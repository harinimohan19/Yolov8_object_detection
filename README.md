# YOLOv8 Object Detection APP

## Project Overview
This project demonstrates an end-to-end Object Detection application using YOLOv8 model. Users can upload an image via a Streamlit web interface, and the backend FastAPI inference server processes the request using a pretrained YOLOv8n model. Detected objects are drawn on the image along with confidence scores and bounding boxes.

## Commands to Run the Application
- Clone the github repository
```
git clone https://github.com/harinimohan19/Yolov8_object_detection.git
cd Yolov8_object_detection
```

- Build Docker Image
```
docker build -t yolo-fastapi .
```

- Run the Container
```
docker run -p 8000:8000 -p 8501:8501 yolo-fastapi
```

Open Streamlit (Windows) → http://localhost:8501