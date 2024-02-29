from deepface import DeepFace
from deepface.modules import verification
import os
from fastapi import FastAPI, File, UploadFile, HTTPException,Request
from fastapi.responses import HTMLResponse,RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import pandas as pd
import numpy as np
import cv2

templates = Jinja2Templates(directory="view")

app = FastAPI()

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

# 디렉터리 생성 확인
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

def generate_heatmap(euclidean_distance_vector, filename):
    # 유클리디안 거리 벡터를 히트맵으로 변환 및 저장
    size = int(np.ceil(np.sqrt(len(euclidean_distance_vector))))
    padded_vector = np.pad(euclidean_distance_vector, (0, size*size - len(euclidean_distance_vector)), 'constant', constant_values=(0, 0))
    vector_2d = padded_vector.reshape((size, size))
    normalized_vector = cv2.normalize(vector_2d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_img = cv2.applyColorMap(normalized_vector, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(temp_dir, filename)
    cv2.imwrite(heatmap_path, heatmap_img)
    return heatmap_path

def overlay_heatmap_on_image(original_image_path, heatmap_path, output_filename, alpha=1.0):
    original_image = cv2.imread(original_image_path)
    heatmap = cv2.imread(heatmap_path)
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    overlayed_image = cv2.addWeighted(original_image, 1-alpha, heatmap_resized, alpha, 0)
    output_path = os.path.join(temp_dir, output_filename)
    cv2.imwrite(output_path, overlayed_image)
    return output_path                     

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
        
        original_image = cv2.imread(temp_file_path)
        # print("00. original_image",original_image)
        original_shape = original_image.shape[:2]  # (height, width)
        # print("00. original_shape",original_shape)
        
        # dfs가 리스트이고 리스트의 첫 번째 요소가 DataFrame인지 확인
        if isinstance(dfs, list) and len(dfs) > 0 and isinstance(dfs[0], pd.DataFrame):
            df = dfs[0]  # DataFrame 추출
            
            print("df",df)
            
            if not df.empty:
            # 'distance' 기준으로 정렬 후 상위 5개 선택
                top_similar_faces = df.sort_values(by="distance", ascending=True).head(5)
            # 결과 정보 추출
                paths = top_similar_faces['identity'].tolist()  # 이미지 경로 리스트로 변환
                distances = top_similar_faces['distance'].tolist()  # 거리 값 리스트로 변환
                heatmap_paths = [] # 히트맵 경로 초기화

                for index, row in top_similar_faces.iterrows():
                    identity_path = row['identity']
                    # DeepFace.represent를 사용하여 표현 벡터 추출
                    db_result = DeepFace.represent(img_path=row['identity'], model_name=models[2], detector_backend=backends[2], enforce_detection=True)
                    target_result = DeepFace.represent(img_path=temp_file_path, model_name=models[2], detector_backend=backends[2], enforce_detection=True)
                    
                    # 임베딩 벡터 추출
                    db_representation = db_result[0]['embedding']
                    target_representation = target_result[0]['embedding']
                    
                    # print("1. db_representation",db_representation)
                    # print("2. db_representation type",type(db_representation))
                    # print("3. target_representation",target_representation)
                    # print("4. target_representation type",type(target_representation))

                    # 유클리디안 거리 계산
                    euclidean_distance_vector = np.subtract(target_representation, db_representation)
                    # euclidean_distance_squared = np.sum(np.square(euclidean_distance_vector))
                    euclidean_distance_squared = np.multiply(euclidean_distance_vector,euclidean_distance_vector)
                    
                    # print("len(euclidean_distance_vector)",len(euclidean_distance_vector))
                    
                    # print("5. euclidean_distance_vector",db_representation)
                    # print("6. euclidean_distance_vector type",type(db_representation))
                    # print("7. euclidean_distance_squared",target_representation)
                    # print("8. euclidean_distance_squared type",type(target_representation))

                    # 히트맵 생성 및 저장
                    heatmap_filename = f"heatmap_{index}.png"
                    heatmap_path = generate_heatmap(euclidean_distance_squared,heatmap_filename)
                    heatmap_paths.append(heatmap_path)
                    
                    output_filename = f"overlayed_{index}.png"  # 오버레이된 이미지의 파일명
                    overlay_path = overlay_heatmap_on_image(identity_path, heatmap_path, output_filename, alpha=0.6)
                    print(f"Overlayed image saved to: {overlay_path}")

                    # print("Top similar faces paths:", paths)
                    # print("Distances:", distances)
                    # print("Heatmap paths:", heatmap_paths)
            
                # 결과 반환
                return {"top_similar_faces_paths": paths, "distances": distances, "heatmap_paths": heatmap_paths}
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