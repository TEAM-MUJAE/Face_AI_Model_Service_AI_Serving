from fastapi import FastAPI, File, UploadFile, Request, APIRouter
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse
from deepface.detectors import DetectorWrapper
from tempfile import NamedTemporaryFile
from deepface import DeepFace
import numpy as np
import cv2
import io
import os
import dlib
import math
import base64
from sift.sift import get_sift_features

router = APIRouter()
templates = Jinja2Templates(directory="view")

# 랜드마크 모델 로드
predictor_path = "models/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# 디렉터리 생성 확인
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# 이미지에서 얼굴 랜드마크를 찾고, 그린 후 파일로 저장하는 함수
def save_landmarked_images_with_sift(image_path1,image_path2, detector, predictor, output_dir, filename_prefix):
    image = cv2.imread(image_path1)
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
        
        compare_image = cv2.resize(compare_image, (base_image.shape[1], base_image.shape[0]))
        # gray_compare = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)
        # faces_compare = detector(gray_compare)
        faces_compare = detector(compare_image)
        
        if len(faces_compare) == 0:
            print(f"No faces detected in image {idx}.")
            continue
        
        # landmarks_compare = predictor(gray_compare, faces_compare[0])
        landmarks_compare = predictor(compare_image, faces_compare[0])
        feature_boxes_compare = calculate_feature_boxes(landmarks_compare)  # 비교 이미지의 바운더리 박스 계산
        draw_feature_boxes(compare_image, feature_boxes_compare)  # 비교 이미지에 바운더리 박스를 그림
        compare_filtered_kps, compare_filtered_descs = get_filtered_keypoints_and_descriptors(compare_image, sift, feature_boxes_compare)  # 필터링된 특징점 추출

        # FLANN 매처 사용하여 매칭
        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(base_filtered_descs, compare_filtered_descs, k=2)

        # 좋은 매치 필터링
        good_matches = [m for m, n in matches if m.distance <0.8 * n.distance]

        # 매칭 결과 시각화
        matched_img = cv2.drawMatches(base_image, base_filtered_kps, compare_image, compare_filtered_kps, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # 결과 이미지 저장
        output_path = os.path.join(output_dir, f"{filename_prefix}_{idx}_matched.jpg")
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
        
        compare_image = cv2.resize(compare_image, (base_image.shape[1], base_image.shape[0]))
        gray_compare = cv2.cvtColor(compare_image, cv2.COLOR_BGR2GRAY)
        faces_compare = detector(gray_compare)
        landmarks_compare = predictor(gray_compare, faces_compare[0])
        feature_boxes_compare = calculate_feature_boxes(landmarks_compare)
        compare_filtered_kps, compare_filtered_descs = get_filtered_keypoints_and_descriptors(compare_image, sift, {feature: feature_boxes_compare[feature]})

        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(base_filtered_descs, compare_filtered_descs, k=2)

        # 좋은 매치 필터링하여 평균 거리 계산
        good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]
        if good_matches:
            average_distance = math.sqrt(sum(m.distance for m in good_matches) / len(good_matches))
        else:
            # average_distance = float('inf')  # 매치가 없는 경우, 평균 거리를 무한대로 설정
            average_distance = 100000000

        compare_scores.append((path, average_distance))

    # 평균 거리(유사도 점수)에 따라 순위를 매기고 반환 (값이 작을수록 더 유사)
    ranked_scores = sorted(compare_scores, key=lambda x: x[1])

    return ranked_scores


# def show_image(img, window_name):
#     cv2.imshow(window_name, img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

@router.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 임시 파일로 저장한 이미지를 base64로 인코딩하여 상대 경로로 변환하는 함수
def get_encoded_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode("utf-8")
    return encoded_string
    
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
    faces3 = DetectorWrapper.detect_faces(img=img3, detector_backend='dlib')
    
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
        
        pw = int(w * 0.2)
        ph = int(h * 0.2)
        
        x = max(0, x- pw)
        y = max(0, y- ph)
        w = w + pw*2
        h = h + ph*2
        
        return img[y:y+h, x:x+w]
    
    cropped_face1 = facial_area(img1, faces1[0])
    cropped_face2 = facial_area(img2, faces2[0])
    cropped_face3 = facial_area(img3, faces3[0])
    


    # 크롭된 얼굴 이미지와 결과를 반환
    _, encoded_img1 = cv2.imencode('.png', cropped_face1)
    _, encoded_img2 = cv2.imencode('.png', cropped_face2)
    _, encoded_img3 = cv2.imencode('.png', cropped_face3)
    encoded_img_bytes1 = encoded_img1.tobytes()
    encoded_img_bytes2 = encoded_img2.tobytes()
    encoded_img_bytes3 = encoded_img3.tobytes()
    
    # 크롭된 얼굴 이미지 표시
    # show_image(cropped_face1, "Cropped Face 1")
    # show_image(cropped_face2, "Cropped Face 2")
    # show_image(cropped_face3, "Cropped Face 3")


    try:
        print("A1")
        
        temp_file_path1 = os.path.join(temp_dir, file1.filename)
        
        print("B1")
        
        temp_file_path2 = os.path.join(temp_dir, file2.filename)
        
        print("B2")

        temp_file_path3 = os.path.join(temp_dir, file3.filename)
        
        print("B3")

        # 임시 파일로 저장
        with NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".png") as temp_file1:
            temp_file1.write(cv2.imencode('.png', cropped_face1)[1])
            temp_file_path1 = os.path.relpath(temp_file1.name, start=os.getcwd())
        
        print("C1")
        
        with NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".png") as temp_file2:
            temp_file2.write(cv2.imencode('.png', cropped_face2)[1])
            temp_file_path2 = os.path.relpath(temp_file2.name, start=os.getcwd())
            
        print("C2")     

        with NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".png") as temp_file3:
            temp_file3.write(cv2.imencode('.png', cropped_face3)[1])
            temp_file_path3 = os.path.relpath(temp_file3.name, start=os.getcwd())
        
        print("C3")
        
        # 이미지를 base64로 인코딩하여 상대 경로로 출력
        encoded_img1 = get_encoded_image(temp_file_path1)
        encoded_img2 = get_encoded_image(temp_file_path2)    
        encoded_img3 = get_encoded_image(temp_file_path3)    
            
        # filename_prefix1 = os.path.splitext(os.path.basename(temp_file_path1))[0]
        # filename_prefix2 = os.path.splitext(os.path.basename(temp_file_path2))[0]
        # filename_prefix3 = os.path.splitext(os.path.basename(temp_file_path3))[0]

        print("D1")
        
        # 특징 매칭 및 시각화
        sift = cv2.SIFT_create()
        landmark_sift_paths = []
        left_eye_similarity = []
        right_eye_similarity = []
        nose_similarity = []
        mouth_similarity = []
        
        print("D2")
        
        # 파일 1과 파일 2 간의 매칭 및 시각화
        landmark_sift_paths1 = match_and_visualize_sift_features(temp_file_path1, [temp_file_path2], detector, predictor, sift, temp_dir, os.path.splitext(os.path.basename(temp_file_path2))[0])
        if landmark_sift_paths1:
            landmark_sift_paths.extend(landmark_sift_paths1)

        print("E1")

        # 파일 1과 파일 3 간의 매칭 및 시각화
        landmark_sift_paths2 = match_and_visualize_sift_features(temp_file_path1, [temp_file_path3], detector, predictor, sift, temp_dir, os.path.splitext(os.path.basename(temp_file_path3))[0])
        if landmark_sift_paths2:
            landmark_sift_paths.extend(landmark_sift_paths2)

        print("E2")
                
        for temp_file_path in [temp_file_path2, temp_file_path3]:
            print(temp_file_path)
            # landmark_sift_path = match_and_visualize_sift_features(temp_file_path1, [temp_file_path], detector, predictor, sift, temp_dir, os.path.splitext(os.path.basename(temp_file_path1))[0])
            # landmark_sift_paths.append(landmark_sift_path)

            print("F1")
            
            left_eye_similarity.append(calculate_feature_similarity(temp_file_path1, [temp_file_path], 'left_eye', detector, predictor, sift, temp_dir))
            right_eye_similarity.append(calculate_feature_similarity(temp_file_path1, [temp_file_path], 'right_eye', detector, predictor, sift, temp_dir))
            nose_similarity.append(calculate_feature_similarity(temp_file_path1, [temp_file_path], 'nose', detector, predictor, sift, temp_dir))
            mouth_similarity.append(calculate_feature_similarity(temp_file_path1, [temp_file_path], 'mouth', detector, predictor, sift, temp_dir))
            
        # 첫 번째 이미지와 두 번째 이미지 비교
        result1 = DeepFace.verify(temp_file_path1, temp_file_path2, model_name='GhostFaceNet', detector_backend='dlib', distance_metric='cosine')
        
        # similarity_percent1 = int((1 - (result1['distance']/result1['threshold'])) * 100)
        
        print("A2")

        # 첫 번째 이미지와 세 번째 이미지 비교
        result2 = DeepFace.verify(temp_file_path1, temp_file_path3, model_name='GhostFaceNet', detector_backend='dlib', distance_metric='cosine')
        
        # similarity_percent2 = int((1 - (result2['distance']/result2['threshold'])) * 100)          
        
        print("A3")


        # 결과 반환
        return {
            "landmark_sift_paths": landmark_sift_paths,
            "left_eye_similarity": left_eye_similarity,
            "right_eye_similarity": right_eye_similarity,
            "nose_similarity": nose_similarity,
            "mouth_similarity": mouth_similarity,
            # "face1": FileResponse(io.BytesIO(encoded_img_bytes1), media_type='image/png'),
            # "face2": FileResponse(io.BytesIO(encoded_img_bytes2), media_type='image/png'),
            # "face3": FileResponse(io.BytesIO(encoded_img_bytes3), media_type='image/png'),
            "encoded_img1": encoded_img1, 
            "encoded_img2": encoded_img2,
            "encoded_img3": encoded_img3,
            "result1": result1, 
            "result2": result2
        }
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)