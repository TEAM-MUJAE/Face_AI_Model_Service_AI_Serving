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

@router.post("/other")
async def verify_other(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        contents1 = await file1.read()
        contents2 = await file2.read()

        nparr1 = np.frombuffer(contents1, np.uint8)
        nparr2 = np.frombuffer(contents2, np.uint8)

        img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)


        # 얼굴 감지
        faces1 = DetectorWrapper.detect_faces(img=img1, detector_backend='dlib')
        faces2 = DetectorWrapper.detect_faces(img=img2, detector_backend='dlib')
        
        # 각 이미지에서 감지된 얼굴 수 확인
        if len(faces1) > 1 or len(faces2) > 1:
            return {"error": "한 이미지에 두 명 이상의 인물이 검출되었습니다."}
        
        # cropped_faces = []

        # # img1에서 감지된 얼굴 크롭
        # for detected_face in faces1:
        #     # 얼굴 위치 정보 추출 방식을 `DetectedFace` 객체의 실제 구조에 맞게 수정
        #     x, y, w, h = detected_face['x'], detected_face['y'], detected_face['w'], detected_face['h']
        #     cropped_face = img1[y:y+h, x:x+w]
        #     cropped_faces.append(cropped_face)

        # # img2에서 감지된 얼굴 크롭
        # for detected_face in faces2:
        #     # 얼굴 위치 정보 추출 방식을 `DetectedFace` 객체의 실제 구조에 맞게 수정
        #     x, y, w, h = detected_face['x'], detected_face['y'], detected_face['w'], detected_face['h']
        #     cropped_face = img2[y:y+h, x:x+w]
        #     cropped_faces.append(cropped_face)


        # 이미지 비교
        result = DeepFace.verify(img1, img2, model_name='Facenet512', detector_backend='dlib', distance_metric='euclidean')
        return result

    except Exception as e:
        return {"error": str(e)}
