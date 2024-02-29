from fastapi import FastAPI, File, UploadFile, Request, APIRouter
from fastapi.templating import Jinja2Templates
from deepface.detectors import DetectorWrapper
from deepface import DeepFace
import numpy as np
import cv2

app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="view")

@router.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.post("/people")
async def compare_faces(file1: UploadFile = File(...), file2: UploadFile = File(...), file3: UploadFile = File(...)):
     
    contents1 = await file1.read()
    contents2 = await file2.read()
    contents3 = await file3.read()

    nparr1 = np.frombuffer(contents1, np.uint8)
    nparr2 = np.frombuffer(contents2, np.uint8)
    nparr3 = np.frombuffer(contents3, np.uint8)

    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)
    img3 = cv2.imdecode(nparr3, cv2.IMREAD_COLOR)
   
    faces1 = DetectorWrapper.detect_faces(img=img1, detector_backend='dlib')
    faces2 = DetectorWrapper.detect_faces(img=img2, detector_backend='dlib')
    faces3 = DetectorWrapper.detect_faces(img=img2, detector_backend='dlib')
    
    if len(faces1) > 1 or len(faces2) > 1 or len(faces3) > 1:
            return {"error": "한 이미지에 두 명 이상의 인물이 검출되었습니다."}


    try:
        # 첫 번째 이미지와 두 번째 이미지 비교
        result1 = DeepFace.verify(img1, img2, model_name='Facenet512', detector_backend='dlib', distance_metric='euclidean')
        
        # similarity_percent1 = int((1 - (result1['distance']/result1['threshold'])) * 100)

        # 첫 번째 이미지와 세 번째 이미지 비교
        result2 = DeepFace.verify(img1, img3, model_name='Facenet512', detector_backend='dlib', distance_metric='euclidean')
        
        # similarity_percent2 = int((1 - (result2['distance']/result2['threshold'])) * 100)          
        

        return {"result1": result1, "result2": result2}

    
    except Exception as e:
        return {"error": str(e)}
