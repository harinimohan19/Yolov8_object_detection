from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import json
import cv2
from PIL import Image
from ultralytics import YOLO

app = FastAPI(title="Simple YOLOv8n FastAPI Backend")

MODEL = YOLO("yolov8n.pt")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    if not file.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Upload an image file.")

    img_path = UPLOAD_DIR / file.filename
    with open(img_path, "wb") as f:
        f.write(await file.read())


    results = MODEL.predict(
        source=str(img_path),
        imgsz=640,
        conf=0.25,
        device="cpu",
        verbose=False
    )

    res = results[0]
    names = MODEL.names

    detections = []
    for box in res.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        detections.append({
            "class_id": cls_id,
            "class_name": names[cls_id],
            "confidence": round(conf, 4),
            "bbox_xyxy": [round(float(v), 2) for v in xyxy]
        })


    json_name = f"{img_path.stem}.json"
    json_path = OUTPUT_DIR / json_name
    with open(json_path, "w") as jf:
        json.dump({
            "input_image": file.filename,
            "detections": detections
        }, jf, indent=4)

    annotated = res.plot()                         
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated_img = Image.fromarray(annotated_rgb)

    annotated_name = f"{img_path.stem}_annotated.jpg"
    annotated_path = OUTPUT_DIR / annotated_name
    annotated_img.save(annotated_path)

    return {
        "message": "success",
        "json_file": json_name,
        "annotated_image": annotated_name,
        "detections": detections
    }


@app.get("/outputs/{file_name}")
def get_output_file(file_name: str):
    """Serve saved files."""
    file_path = OUTPUT_DIR / file_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
