from deepface import DeepFace
import os
from fastapi import FastAPI, File, UploadFile, HTTPException,Request
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import pandas as pd

templates = Jinja2Templates(directory="view")

app = FastAPI()

# 디렉터리 생성 확인
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

models = [
    "VGG-Face", 
    "Facenet", 
    "Facenet512", 
    "OpenFace", 
    "DeepFace", 
    "DeepID", 
    "ArcFace", 
    "Dlib", 
    "SFace",
]

backends = [
    'opencv', 
    'ssd', 
    'dlib', 
    'mtcnn', 
    'retinaface', 
    'mediapipe',
    'yolov8',
    'yunet',
    'fastmtcnn',
]

metrics = ["cosine", "euclidean", "euclidean_l2"]

db_path = r"C:\img_for_ai"  # 연예인 데이터 경로

@app.get("/",response_class=HTMLResponse)
async def main(request : Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    
    temp_file_path = os.path.join(temp_dir, file.filename)

    # print("start")
    # print("temdir",temp_dir)
    # print("temdir",file.filename)

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # print("tempfilepath", temp_file_path)    
    
        
    try:
        # DeepFace.find 호출하여 유사한 얼굴 찾기
        dfs = DeepFace.find(img_path=temp_file_path,
                            db_path=db_path,
                            model_name=models[2],  # Facenet512
                            distance_metric=metrics[1],  # euclidean
                            detector_backend=backends[2],  # dlib
                            threshold = 1000
                            )        

        # print("Dfs",dfs)
        
        # dfs가 리스트이고 리스트의 첫 번째 요소가 DataFrame인지 확인
        if isinstance(dfs, list) and len(dfs) > 0 and isinstance(dfs[0], pd.DataFrame):
            df = dfs[0]  # DataFrame 추출
            if not df.empty:
            # 'distance' 기준으로 정렬 후 상위 5개 선택
                top_similar_faces = df.sort_values(by="distance", ascending=True).head(5)
            # 결과 정보 추출
                paths = top_similar_faces['identity'].tolist()  # 이미지 경로 리스트로 변환
                distances = top_similar_faces['distance'].tolist()  # 거리 값 리스트로 변환

            # 결과 반환
                return {"top_similar_faces_paths": paths, "distances": distances}
            else:
                return {"message": "No similar faces found."}
        else:
        # 'dfs'가 예상한 형식이 아닐 경우 오류 메시지 반환
            return {"error": "Unexpected result format. Expected a DataFrame inside a list."}
    
    except Exception as e:
        return {"error": str(e)}

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    uvicorn.run("comp_celeb:app", port=8000,reload=True)