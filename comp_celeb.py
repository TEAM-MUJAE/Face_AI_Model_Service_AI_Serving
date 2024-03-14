from deepface import DeepFace
from deepface.detectors import DetectorWrapper
from fastapi import FastAPI, File, UploadFile,Request,APIRouter #,HTTPException
from fastapi.responses import HTMLResponse,JSONResponse #,RedirectResponse
from fastapi.templating import Jinja2Templates
from sift.sift import get_sift_features

import os
import shutil
import pandas as pd
import numpy as np
import cv2
import dlib
import math
import base64

router = APIRouter()

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

# 연예인 데이터 디렉토리 이름
db_dirname = "img_for_ai"

# 운영 체제에 따라 최상위 디렉토리 설정
if os.name == "nt":  # Windows
    top_dir = "C:/"
elif os.name == "posix":  # macOS or Linux
    top_dir = "/"

print("top_dir",top_dir)

# 연예인 데이터 경로
db_path = os.path.join(top_dir, db_dirname)

print("db_path",db_path)

predictor_path = "models/shape_predictor_68_face_landmarks.dat"  # 랜드마크 모델 파일 경로
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# 디렉터리 생성 확인
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
    
# 얼굴 크롭 및 저장
def facial_area(img, detected_face):
    # 이 객체는 x, y, w, h 속성을 가진다.
    x = detected_face.facial_area.x
    y = detected_face.facial_area.y
    w = detected_face.facial_area.w
    h = detected_face.facial_area.h
    return img[y:y+h, x:x+w]

# 이미지에서 얼굴 랜드마크를 찾고, 그린 후 파일로 저장하는 함수
def save_landmarked_images_with_sift(image_path, detector, predictor, output_dir, filename_prefix, distances):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = detector(gray)
    if len(faces) == 0:
        return None
    
    face = faces[0]
    landmarks = predictor(gray, face)
    
    # SIFT 적용
    keypoints, descriptors = get_sift_features(image)
    
    # 각 얼굴 특징(눈, 코, 입)에 대한 인덱스 및 색상 정의
    features_indices = {
        "left_eye": (range(36, 42), (255, 0, 0),5),
        "right_eye": (range(42, 48), (0, 255, 0),5),
        "nose": (range(27, 36), (0, 0, 255),0),
        "mouth": (range(48, 68), (255, 255, 0),0)
    }
    
    for feature, (indices, color, padding) in features_indices.items():
        points = np.array([[p.x, p.y] for p in landmarks.parts()])[list(indices)]
        
        x_min, y_min = np.min(points, axis=0) - padding
        x_max, y_max = np.max(points, axis=0) + padding
        
        # 각 특징의 바운더리 박스 그리기
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        
        # 바운더리 박스 내 SIFT 특징점만 필터링
        feature_keypoints = [kp for kp in keypoints if x_min <= kp.pt[0] <= x_max and y_min <= kp.pt[1] <= y_max]

        # SIFT 특징점에 색상 표시
        for kp in feature_keypoints:
            cv2.circle(image, (int(kp.pt[0]), int(kp.pt[1])), 3, color, 2)
        
    # 시각화된 이미지 저장
    output_path = os.path.join(output_dir, f"{filename_prefix}_landmarked_with_sift.jpg")
    cv2.imwrite(output_path, image)
    return output_path

# 랜드마크 포인트에 따른 바운더리 박스 계산 함수
def calculate_feature_boxes(landmarks):
    
    # 각 얼굴 특징(눈, 코, 입)에 대한 인덱스 및 색상 정의
    features_indices = {
        "left_eye": (range(36, 42), (255, 0, 0),5),
        "right_eye": (range(42, 48), (0, 255, 0),5),
        "nose": (range(27, 36), (0, 0, 255),0),
        "mouth": (range(48, 68), (255, 255, 0),0)
    }
    
    feature_boxes = {}
    for feature, (indices, _, padding) in features_indices.items():
        points = np.array([[landmarks.part(n).x, landmarks.part(n).y] for n in indices])
        x_min, y_min = np.min(points, axis=0) - padding
        x_max, y_max = np.max(points, axis=0) + padding
        feature_boxes[feature] = (x_min, y_min, x_max, y_max)
    return feature_boxes

# 이미지에서 SIFT 특징점과 설명자를 추출하고 바운더리 박스에 해당하는 특징점만 필터링
def get_filtered_keypoints_and_descriptors(image, sift, feature_boxes):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    filtered_keypoints = []
    filtered_descriptors = []
    for kp, desc in zip(keypoints, descriptors):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        for box in feature_boxes.values():
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                filtered_keypoints.append(kp)
                filtered_descriptors.append(desc)
                break
    return filtered_keypoints, np.array(filtered_descriptors)


def draw_feature_boxes(image, feature_boxes):
    """
    이미지에 랜드마크 기반 바운더리 박스를 그립니다.
    """
    # for feature, box in feature_boxes.items():
    #     cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
    return image


def match_and_visualize_sift_features(base_image_path, compare_image_paths, detector, predictor, sift, output_dir,filename_prefix):
    """
    기준 이미지와 비교 이미지 간의 SIFT 특징점을 매칭하고 시각화합니다.
    결과로 5장의 이미지가 생성되며, 각 이미지는 기준 이미지와 한 비교 이미지 사이의 매칭을 보여줍니다.
    """

    # 결과 이미지 경로들을 저장할 리스트
    encoded_images = []
    
    base_image = cv2.imread(base_image_path)
    gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    faces_base = detector(gray_base)
    if not faces_base:
        print("No faces detected in the base image.")
        return
    
    landmarks_base = predictor(gray_base, faces_base[0])
    feature_boxes_base = calculate_feature_boxes(landmarks_base)  # 기준 이미지의 바운더리 박스 계산
    base_image_with_boxes = draw_feature_boxes(base_image.copy(), feature_boxes_base)
    base_filtered_kps, base_filtered_descs = get_filtered_keypoints_and_descriptors(base_image_with_boxes, sift, feature_boxes_base)  # 필터링된 특징점 추출


    for idx, compare_image_path in enumerate(compare_image_paths):
        compare_image = cv2.imread(compare_image_path)
        gray_compare = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)
        faces_compare = detector(gray_compare)
        
        if len(faces_compare) == 0:
            print(f"No faces detected in image {idx}.")
            continue
        
        landmarks_compare = predictor(gray_compare, faces_compare[0])
        feature_boxes_compare = calculate_feature_boxes(landmarks_compare)  # 비교 이미지의 바운더리 박스 계산
        draw_feature_boxes(compare_image, feature_boxes_compare)  # 비교 이미지에 바운더리 박스를 그림
        compare_filtered_kps, compare_filtered_descs = get_filtered_keypoints_and_descriptors(compare_image, sift, feature_boxes_compare)  # 필터링된 특징점 추출

        # FLANN 매처 사용하여 매칭
        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(base_filtered_descs, compare_filtered_descs, k=2)

        # 좋은 매치 필터링
        good_matches = [m for m, n in matches if m.distance <1 * n.distance]

        # 매칭 결과 시각화
        matched_img = cv2.drawMatches(base_image, base_filtered_kps, compare_image, compare_filtered_kps, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # 결과 이미지 저장
        output_path = os.path.join(output_dir, f"{filename_prefix}_matched.jpg")
        cv2.imwrite(output_path, matched_img)
        print(f"Saved matched image to {output_path}")
        
        # 결과 이미지를 base64로 인코딩
        retval, buffer = cv2.imencode('.jpg', matched_img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        image_name = f"{filename_prefix}"  # 이미지 이름 생성
        print(f"image_name",image_name)
        # print(f"Encoded matched image to base64",encoded_image)
        encoded_images.append((image_name, encoded_image))
        
    return encoded_images

# # 바운더리 박스 내의 특징점만 필터링하는 함수
# def filter_keypoints_by_feature_boxes(keypoints, descriptors, feature_boxes):
#     filtered_keypoints = []
#     filtered_descriptors = []
#     for kp, desc in zip(keypoints, descriptors):
#         x, y = kp.pt
#         for box in feature_boxes.values():
#             x_min, y_min, x_max, y_max = box
#             if x_min <= x <= x_max and y_min <= y <= y_max:
#                 filtered_keypoints.append(kp)
#                 filtered_descriptors.append(desc)
#                 break
#     for kp, desc in zip(keypoints, descriptors):
#         x, y = kp.pt
#         for box in feature_boxes.values():
#             x_min, y_min, x_max, y_max = box
#             if x_min <= x <= x_max and y_min <= y <= y_max:
#                 filtered_keypoints.append(kp)
#                 filtered_descriptors.append(desc)
#                 break
#     return filtered_keypoints, np.array(filtered_descriptors)

def calculate_feature_similarity(base_image_path, compare_image_paths, feature, detector, predictor, sift, output_dir):
    """
    주어진 특징(눈, 코, 입)에 대해 기준 이미지와 비교 이미지들 간의 평균 거리 기반 유사도 점수를 계산하고 순위를 매깁니다.
    """
    base_image = cv2.imread(base_image_path)
    gray_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    faces_base = detector(gray_base)
    landmarks_base = predictor(gray_base, faces_base[0])
    feature_boxes_base = calculate_feature_boxes(landmarks_base)
    base_filtered_kps, base_filtered_descs = get_filtered_keypoints_and_descriptors(base_image, sift, {feature: feature_boxes_base[feature]})
    
    compare_scores = []
    for path in compare_image_paths:
        compare_image = cv2.imread(path)
        gray_compare = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)
        faces_compare = detector(gray_compare)
        landmarks_compare = predictor(gray_compare, faces_compare[0])
        feature_boxes_compare = calculate_feature_boxes(landmarks_compare)
        compare_filtered_kps, compare_filtered_descs = get_filtered_keypoints_and_descriptors(compare_image, sift, {feature: feature_boxes_compare[feature]})

        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(base_filtered_descs, compare_filtered_descs, k=2)
        
        # 좋은 매치 필터링하여 평균 거리 계산
        good_matches = [m for m, n in matches if m.distance < 1 * n.distance]
        if good_matches:
            average_distance = math.sqrt(sum(m.distance for m in good_matches) / len(good_matches))
        else:
            average_distance = float('inf')  # 매치가 없는 경우, 평균 거리를 무한대로 설정
        
        compare_scores.append((path, average_distance))

    # 평균 거리(유사도 점수)에 따라 순위를 매기고 반환 (값이 작을수록 더 유사)
    ranked_scores = sorted(compare_scores, key=lambda x: x[1])

    return ranked_scores

# @router.get("/",response_class=HTMLResponse)
# async def main(request : Request):
#     return templates.TemplateResponse("index.html", {"request": request})

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    
    temp_file_path = os.path.join(temp_dir, file.filename)

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # print("tempfilepath", temp_file_path)    
    
        
    try:
        # 각 이미지에서 2개 이상 얼굴 감지 시 예외 처리
        content = cv2.imread(temp_file_path)
        
        # print("content is",content)
        
        # 얼굴 감지
        face = DetectorWrapper.detect_faces(img=content, detector_backend='dlib')
        
        # 각 이미지에서 감지된 얼굴 수 확인
        if len(face) > 1 :
            return JSONResponse(content={"error": "한 이미지에 두 명 이상의 인물이 검출되었습니다."}, status_code=400)

        cropped_face = facial_area(content, face[0])
        
        
        # DeepFace.find 호출하여 유사한 얼굴 찾기
        dfs = DeepFace.find(
                            img_path= cropped_face,
                            db_path=db_path,
                            model_name=models[2],  # Facenet512
                            distance_metric=metrics[1],  # euclidean
                            detector_backend=backends[2],  # dlib
                            threshold = 1000000000
                            )        

        # print("Dfs",dfs)
        
        # dfs가 리스트이고 리스트의 첫 번째 요소가 DataFrame인지 확인
        if isinstance(dfs, list) and len(dfs) > 0 and isinstance(dfs[0], pd.DataFrame):
            df = dfs[0]  # DataFrame 추출
            
            # print("df",df)
            
            if not df.empty:
            # 'distance' 기준으로 정렬 후 상위 5개 선택
                top_similar_faces = df.sort_values(by="distance", ascending=True).head(5)
            # 결과 정보 추출
                paths = top_similar_faces['identity'].tolist()  # 이미지 경로 리스트로 변환
                distances = top_similar_faces['distance'].tolist()  # 거리 값 리스트로 변환
                
                total_similar_faces = []
                cropped_file_paths = [] # 크롭된 이미지 경로 초기화
                landmark_paths = [] # 랜드마크된 이미지 경로 초기화
                landmark_sift_paths= [] # 랜드마크된 이미지 값 초기화
                left_eyes_socore_distances= [] # 눈,코,입 유사도 값 초기화
                right_eyes_socore_distances= [] # 눈,코,입 유사도 값 초기화
                nose_socore_distances= [] # 눈,코,입 유사도 값 초기화
                mouth_socore_distances= [] # 눈,코,입 유사도 값 초기화
                
                for path, distance in zip(paths, distances):
                    total_similar_faces.append([path, distance])
                
                # 결과 출력
                print(total_similar_faces)
                
                print("cropped_face",cropped_face)
                
                # 크롭된 이미지 저장
                cropped_filename1 = f"cropped_{file.filename}"  # 크롭된 이미지 파일명 정의
                cropped_filename2 = f"{file.filename}"  # 크롭된 이미지 파일명 정의
                cropped_file_path1 = os.path.join(temp_dir, cropped_filename1)  # 저장 경로 조합
                cropped_file_path2 = os.path.join(temp_dir, cropped_filename2)  # 저장 경로 조합
                cv2.imwrite(cropped_file_path1, cropped_face)  # 이미지 저장
                # cropped_file_paths.append(cropped_file_path)
                
                cropped_filename_prefix = os.path.splitext(os.path.basename(cropped_file_path2))[0]
                
                # 결과 이미지를 base64로 인코딩
                retval, buffer = cv2.imencode('.jpg', cropped_face)
                encoded_cropped_image = base64.b64encode(buffer).decode('utf-8')
                cropped_image_name = f"{cropped_filename_prefix}"  # 이미지 이름 생성
                print(f"cropped_image_name",cropped_image_name)
                                
                cropped_file_paths.append((cropped_image_name, encoded_cropped_image))
                
                for _, row in top_similar_faces.iterrows():
                    identity_path = row['identity']
                    
                    # 각 상위 유사 이미지에 대해 SIFT 적용
                    print("identity_path is",identity_path)
                    
                    filename_prefix = os.path.splitext(os.path.basename(identity_path))[0]
                    print("filename_prefix is",filename_prefix)
                    
                    landmark_path = save_landmarked_images_with_sift(identity_path, detector, predictor, temp_dir, filename_prefix, {})
                    
                    sift = cv2.SIFT_create()
                    
                    landmark_sift_path = match_and_visualize_sift_features(temp_file_path, [identity_path], detector, predictor,sift,temp_dir,filename_prefix)
                    
                    # print("landmark_path is",landmark_path)
                    # print("landmark_sift_path is",landmark_sift_path)
                    
                    if landmark_path:
                        landmark_paths.append(landmark_path)
                        
                    if landmark_sift_path:
                        landmark_sift_paths.append(landmark_sift_path)
                    
                    
                    left_eye_similarity_rankings = calculate_feature_similarity(temp_file_path, [identity_path], 'left_eye', detector, predictor, sift, temp_dir)
                    right_eye_similarity_rankings = calculate_feature_similarity(temp_file_path, [identity_path], 'right_eye', detector, predictor, sift, temp_dir)
                    nose_similarity_rankings = calculate_feature_similarity(temp_file_path, [identity_path], 'nose', detector, predictor, sift, temp_dir)
                    mouth_similarity_rankings = calculate_feature_similarity(temp_file_path, [identity_path], 'mouth', detector, predictor, sift, temp_dir)
                    
                    for ranking in left_eye_similarity_rankings:
                        left_eyes_socore_distances.append(ranking)
                        
                    for ranking in right_eye_similarity_rankings:
                        right_eyes_socore_distances.append(ranking)
                        
                    for ranking in nose_similarity_rankings:
                        nose_socore_distances.append(ranking)
                        
                    for ranking in mouth_similarity_rankings:
                        mouth_socore_distances.append(ranking)
            
                # 결과 반환
                return {
                        "total_similar_faces": total_similar_faces,
                        "cropped_file_paths" : cropped_file_paths,
                        "cropped_file_path1" : cropped_file_path1,
                        # "landmark_paths" : landmark_paths,
                        "landmark_sift_paths" : landmark_sift_paths,
                        "left_eyes_socore_distances" : left_eyes_socore_distances,
                        "right_eyes_socore_distances" : right_eyes_socore_distances,
                        "nose_socore_distances" : nose_socore_distances,
                        "mouth_socore_distances" : mouth_socore_distances,
                        }
            else:
                return {"message": "No similar faces found."}
        else:
        # 'dfs'가 예상한 형식이 아닐 경우 오류 메시지 반환
            return {"error": "Unexpected result format. Expected a DataFrame inside a list."}
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)