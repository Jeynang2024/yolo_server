from fastapi import FastAPI, UploadFile, File
import cv2, numpy as np
from fastapi.responses import StreamingResponse
import io
from ultralytics import YOLO
from fastapi import HTTPException
app = FastAPI()
model = YOLO('best.pt')  # load your custom model

'''@app.post('/detect/')
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(img)
    annotated = results[0].plot()  # or manually draw boxes

    # Encode as PNG
    success, encoded = cv2.imencode('.png', annotated)
    if not success:
        raise HTTPException(status_code=500, detail="Image encoding failed")

    #return StreamingResponse(io.BytesIO(encoded.tobytes()), media_type="image/png")
    detections = [
        {'box': box.tolist(), 'score': float(score), 'class': int(cls)}
        for box, score, cls in zip(
            results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls
        )
    ]
    return {'detections': detections}'''



app = FastAPI()
model = YOLO('best.pt')  # load your custom YOLO model

@app.post('/detect/')
async def detect(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    results = model.predict(source=img)  # run inference
    r = results[0]

    boxes = r.boxes
    # boxes.xyxy returns a tensor of shape (N, 4)
    xyxy = boxes.xyxy.cpu().numpy()  # [[x1, y1, x2, y2], ...]

    confidences = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy()

    detections = [
        {
            'box': box.tolist(),
            'score': float(score),
            'class': int(cls)
        }
        for box, score, cls in zip(xyxy, confidences, classes)
    ]

    return {'detections': detections}


