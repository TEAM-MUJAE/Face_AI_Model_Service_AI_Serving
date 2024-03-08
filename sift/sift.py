import os
import pickle
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import uvicorn

from skimage import io, color
from skimage.feature import local_binary_pattern
from scipy.spatial.distance import euclidean
from itertools import combinations


def getDicFromTxt():
    imageAnnotations = {}
    celebImages = {}

    facesList = os.listdir('faces')

    with open("identity_CelebA.txt", "r") as file:
        for line in file:
            tempLine = line.split(sep=' ')
            tempLine[1] = tempLine[1].rstrip('\n')
            if tempLine[0] in facesList:
                imageAnnotations.update({tempLine[0]: tempLine[1]})
                if tempLine[1] not in celebImages:
                    celebImages.update({tempLine[1]:[tempLine[0]]})
                else:
                    tempList = celebImages.get(tempLine[1])
                    tempList.append(tempLine[0])
                    celebImages.update({tempLine[1]: tempList})

    return imageAnnotations, celebImages

def saveIAandCI(imageanot, celebanot):
    with open('imageAnnotations.txt', 'wb') as fo:
        pickle.dump(imageanot, fo)
    with open('celebImages.txt', 'wb') as fi:
        pickle.dump(celebanot, fi)
    print('SAVING OF IA AND CA WAS SUCCESSFUL')

def loadIAandCI():
    with open('imageAnnotations.txt', 'rb') as fo:
        imageAnnotations = pickle.load(fo)
    with open('celebImages.txt', 'rb') as fi:
        celebImages = pickle.load(fi)
    print('LOADING OF IA AND CA WAS SUCCESSFUL')
    return imageAnnotations, celebImages

def getLBP(color_image_path):
    color_image = cv2.imread(color_image_path, cv2.IMREAD_COLOR)
    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    gray_image = color.rgb2gray(color_image_rgb)
    gray_image = (gray_image * 255).astype('uint8')  # float를 8비트 정수로 변환
    patterns = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
    hist, _ = np.histogram(patterns.ravel(), bins=np.arange(0, patterns.max() + 1))
    hist = hist.astype('float')
    hist /= (hist.sum() + 1e-6)  # 정규화
    return hist

def getSIFT(color_image):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(color_image, None)
    return kp, des


def getLBPandSIFTannotations():
    LBPannotations = {}
    SIFTannotations = {}
    # print("os.listdir('faces') is", os.listdir('faces'))
    for filename in os.listdir('faces'):
        img = cv2.imread('faces/' + filename)
        LBPanon = getLBP(img)
        img = cv2.imread('faces/' + filename, cv2.IMREAD_GRAYSCALE)
        SIFTkp, SIFTdes = getSIFT(img)

        img_SIFT_list = []
        tempSIFTkp = []
        for element in SIFTkp:
            tempList = [element.angle, element.class_id, element.octave, element.pt, element.response, element.size]
            tempSIFTkp.append(tempList)

        img_SIFT_list.append(tempSIFTkp)
        img_SIFT_list.append(SIFTdes)

        # Adding to dictionaries
        LBPannotations.update({str(filename): LBPanon})
        SIFTannotations.update({str(filename): img_SIFT_list})

    return LBPannotations, SIFTannotations


def saveAnnotations(LBPanot, SIFTanot):
    with open('sift/LBPannotations.txt', 'wb') as fo:
        pickle.dump(LBPanot, fo)
    with open('sift/SIFTannotations.txt', 'wb') as fi:
        pickle.dump(SIFTanot, fi)
    print('SAVING OF LBP AND SIFT WAS SUCCESSFUL')


def loadAnnotations():
    with open('sift/LBPannotations.txt', 'rb') as fo:
        LBPanot = pickle.load(fo)
    with open('sift/SIFTannotations.txt', 'rb') as fi:
        SIFTanot = pickle.load(fi)
    print('LOADING OF LBP AND SIFT ANNOTATIONS WAS SUCCESSFUL')


    newSIFTanot = {}
    for i, imageID in enumerate(SIFTanot):
        allKeypoints = []
        imageValue = SIFTanot.get(imageID)
        # print("imageValue is",imageValue)
        imageDes = imageValue[1]
        for j, keypoint in enumerate(imageValue[0]):
            tempKP = cv2.KeyPoint(x=keypoint[3][0], y=keypoint[3][1], size=keypoint[5], angle=keypoint[0],
                                  response=keypoint[4], octave=keypoint[2], class_id=keypoint[1])
            # print("tempKP is",tempKP)
            allKeypoints.append(tempKP)
        newSIFTanot.update({imageID: [allKeypoints, imageDes]})

    return LBPanot, newSIFTanot


# def compareAllImagesWithSIFT(siftannotations, combinations):
#     print("start")
#     flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5), dict())
#     print("flann is",flann)
#     allDistances = []
#     for index, combination in enumerate(combinations):
#         image1 = siftannotations.get(combination[0])
#         kp1 = image1[0]
#         desc1 = image1[1]
#         image2 = siftannotations.get(combination[1])
#         kp2 = image2[0]
#         desc2 = image2[1]

#         matches = flann.knnMatch(desc1, desc2, k=2)

#         good_points = []
#         for m, n in matches:
#             if m.distance < 0.9*n.distance:
#                 good_points.append(m)

#         #Visualize
#         img = cv2.imread('faces/' + combination[0] )
#         imgToCompare = cv2.imread('faces/' + combination[1])
#         result = cv2.drawMatches(img, kp1, imgToCompare, kp2, good_points, None)
#         cv2.imshow('Matches found', cv2.resize(result, None, fx=3, fy=3))
#         cv2.waitKey(0)

#         number_keypoints = 0
#         if len(kp1) <= len(kp2):
#             number_keypoints = len(kp1)
#         else:
#             number_keypoints = len(kp2)

#         howGood = len(good_points) / number_keypoints * 100

#         allDistances.append(howGood)

#     thisReturn = fromDistancesToDivided(allDistances)

#     return thisReturn

def resize_image(image_path, size=(500, 500)):
    """지정된 크기로 이미지를 조정합니다."""
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image

def find_matches_and_visualize_all(test_image_path, sift_annotations):
    """두 이미지 간의 SIFT 특징점 매칭을 찾고 시각화합니다."""
    test_image = resize_image(test_image_path)
    test_kp, test_des = getSIFT(test_image)

    # 모든 비교 이미지에 대하여
    for image_name, (image_kp, image_des) in sift_annotations.items():
        # 비교 이미지를 리사이즈합니다.
        compare_image = resize_image(f'faces/{image_name}')
        # FLANN 매처 설정
        flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
        matches = flann.knnMatch(test_des, np.array(image_des), k=2)

        # 좋은 매치 선별
        good_matches = [m for m, n in matches if m.distance < 1.0*n.distance]

        # 매칭 결과 시각화
        match_image = cv2.drawMatches(test_image, test_kp, compare_image, image_kp, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Matches with {image_name}')
        plt.show()

def celebritySearch(image_path, sift_annotations):
    """테스트 이미지와 가장 유사한 이미지를 찾습니다."""
    test_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(test_image, None)

    best_match = None
    best_good_matches_count = 0
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})

    for filename, (kp2, des2) in sift_annotations.items():
        matches = flann.knnMatch(des, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.7*n.distance]
        if len(good_matches) > best_good_matches_count:
            best_good_matches_count = len(good_matches)
            best_match = filename

    return best_match

def extract_sift_features(image_path):
    # 이미지 로드
    image = cv2.imread(image_path)
    
    # SIFT 객체 생성
    sift = cv2.SIFT_create()
    
    # 흑백 이미지로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 특징점과 설명자 추출
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def get_sift_features(image):
    """
    이미지에서 SIFT를 적용하여 특징점과 설명자를 반환하는 함수
    :param image: 이미지 배열
    :return: 특징점, 설명자 튜플
    """
    # SIFT 객체 생성
    sift = cv2.SIFT_create()
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # SIFT 적용
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


if __name__ == "__main__":
    # imageAnnotations, celebImages = getDicFromTxt()
    # saveIAandCI(imageAnnotations, celebImages)
    imageAnnotations, celebImages = loadIAandCI()
    print("start1")

    # LBPannotations, SIFTannotations = getLBPandSIFTannotations()
    # saveAnnotations(LBPannotations, SIFTannotations)
    LBPannotations, SIFTannotations = loadAnnotations()
    print("start2")

    find_matches_and_visualize_all('test_img.jpg', SIFTannotations)

    # imgJaro = cv2.imread('test_img.jpg')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()