from fastapi import FastAPI, File, UploadFile, Request, APIRouter
from fastapi.responses import FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from deepface.detectors import DetectorWrapper
from deepface import DeepFace
import numpy as np
import cv2
import io

router = APIRouter()
templates = Jinja2Templates(directory="view")


def show_image(img, window_name):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
        
     # 얼굴 크롭 및 저장
    def facial_area(img, detected_face):
        # 가정: detected_face 객체의 facial_area 속성은 FacialAreaRegion 객체이며,
        # 이 객체는 x, y, w, h 속성을 가진다.
        x = detected_face.facial_area.x
        y = detected_face.facial_area.y
        w = detected_face.facial_area.w
        h = detected_face.facial_area.h
        return img[y:y+h, x:x+w]
    
    cropped_face1 = facial_area(img1, faces1[0])
    cropped_face2 = facial_area(img2, faces2[0])
    cropped_face3 = facial_area(img3, faces2[0])
    


    # 크롭된 얼굴 이미지와 결과를 반환
    _, encoded_img1 = cv2.imencode('.png', cropped_face1)
    _, encoded_img2 = cv2.imencode('.png', cropped_face2)
    _, encoded_img3 = cv2.imencode('.png', cropped_face3)
    encoded_img_bytes1 = encoded_img1.tobytes()
    encoded_img_bytes2 = encoded_img2.tobytes()
    encoded_img_bytes3 = encoded_img3.tobytes()
    
    # 크롭된 얼굴 이미지 표시
    show_image(cropped_face1, "Cropped Face 1")
    show_image(cropped_face2, "Cropped Face 2")
    show_image(cropped_face3, "Cropped Face 3")


    try:
        # 첫 번째 이미지와 두 번째 이미지 비교
        result1 = DeepFace.verify(cropped_face1, cropped_face2, model_name='Facenet512', detector_backend='dlib', distance_metric='euclidean')
        
        # similarity_percent1 = int((1 - (result1['distance']/result1['threshold'])) * 100)

        # 첫 번째 이미지와 세 번째 이미지 비교
        result2 = DeepFace.verify(cropped_face1, cropped_face3, model_name='Facenet512', detector_backend='dlib', distance_metric='euclidean')
        
        # similarity_percent2 = int((1 - (result2['distance']/result2['threshold'])) * 100)          
        

        return {
            "face1": FileResponse(io.BytesIO(encoded_img_bytes1), media_type='image/png'),
            "face2": FileResponse(io.BytesIO(encoded_img_bytes2), media_type='image/png'),
            "face3": FileResponse(io.BytesIO(encoded_img_bytes3), media_type='image/png'),
            "result1": result1, 
            "result2": result2}

    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
