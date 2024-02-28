from fastapi import FastAPI, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from deepface import DeepFace
import numpy as np
import cv2

app = FastAPI()

templates = Jinja2Templates(directory="view")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/other")
async def verify_other(file1: UploadFile = File(...), file2: UploadFile = File(...)):
     
    contents1 = await file1.read()
    contents2 = await file2.read()

    nparr1 = np.frombuffer(contents1, np.uint8)
    nparr2 = np.frombuffer(contents2, np.uint8)

    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    try:
        # 이미지 비교
        result = DeepFace.verify(img1, img2, model_name='Facenet512', detector_backend='dlib', distance_metric='euclidean')
        # result['threshold'] = 1000
        
        similarity_percent = int((1 - (result['distance']/result['threshold'])) * 100)
        print(similarity_percent)
        return result
    
    except Exception as e:
        return {"error": str(e)}
    