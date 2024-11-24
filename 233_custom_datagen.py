"""
Custom data generator to work with BraTS2020 dataset.
Can be used as a template to create your own custom data generators. 

No image processing operations are performed here, just load data from local directory
in batches. 

NPY 이해를 못 함 => 변형하는 코드

"""

# 필요한 라이브러리 임포트
import os
import numpy as np

# 이미지를 로드하는 함수
def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):  # 이미지 리스트를 반복
        if (image_name.split('.')[1] == 'npy'):  # 파일 확장자가 .npy인 경우에만 처리
            image = np.load(img_dir + image_name)  # .npy 파일을 NumPy 배열로 로드
            images.append(image)  # 이미지를 리스트에 추가
    images = np.array(images)  # 리스트를 NumPy 배열로 변환
    return images

# 데이터 제너레이터 함수 정의
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    """
    Keras 모델 학습을 위한 사용자 정의 데이터 제너레이터.
    이미지를 배치 단위로 로드하여 반복적으로 반환.

    img_dir: 이미지 디렉토리 경로
    img_list: 이미지 파일 리스트
    mask_dir: 마스크 디렉토리 경로
    mask_list: 마스크 파일 리스트
    batch_size: 배치 크기
    """
    L = len(img_list)  # 총 이미지 개수

    # Keras는 제너레이터가 무한히 작동해야 하므로 while True 사용
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:  # 모든 이미지를 처리할 때까지 반복
            limit = min(batch_end, L)  # 배치 끝이 이미지 개수를 초과하지 않도록 제한

            # 배치 크기만큼 이미지와 마스크를 로드
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X, Y)  # 이미지와 마스크의 튜플 반환

            # 다음 배치를 위한 인덱스 업데이트
            batch_start += batch_size   
            batch_end += batch_size

############################################

# 제너레이터 테스트 코드
from matplotlib import pyplot as plt
import random

# 학습 이미지와 마스크 디렉토리 설정
train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"
train_img_list = os.listdir(train_img_dir)  # 이미지 파일 리스트
train_mask_list = os.listdir(train_mask_dir)  # 마스크 파일 리스트

# 배치 크기 설정
batch_size = 2

# 학습 데이터 제너레이터 생성
train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

# 제너레이터 작동 확인 (Python 3에서 next()는 __next__()로 변경됨)
img, msk = train_img_datagen.__next__()  # 다음 배치의 이미지와 마스크 가져오기

# 랜덤으로 배치 내 이미지 선택
img_num = random.randint(0, img.shape[0]-1)
test_img = img[img_num]  # 선택한 이미지
test_mask = msk[img_num]  # 선택한 마스크
test_mask = np.argmax(test_mask, axis=3)  # 마스크를 클래스 값으로 변환

# 랜덤으로 슬라이스 선택
n_slice = random.randint(0, test_mask.shape[2])

# 선택한 이미지와 마스크를 시각화
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')  # FLAIR 채널 시각화
plt.title('Image flair')

plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')  # T1CE 채널 시각화
plt.title('Image t1ce')

plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')  # T2 채널 시각화
plt.title('Image t2')

plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])  # 마스크 시각화
plt.title('Mask')

plt.show()
