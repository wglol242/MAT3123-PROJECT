import os
import numpy as np
import random

# .npy => Numpy 배열로 변환
def load_img(img_dir, img_list):
    images = []
    for i, image_name in enumerate(img_list):  # 이미지 리스트를 반복
        if (image_name.split('.')[1] == 'npy'):  # 파일 확장자가 .npy인 경우에만 처리
            image = np.load(img_dir + image_name).astype(np.float32)  # .npy 파일을 NumPy 배열로 로드
            images.append(image)  # 이미지를 리스트에 추가
    images = np.array(images)  # 리스트를 NumPy 배열로 변환
    return images

# 이미지와 마스크 데이터를 배치(batch) 단위로 반복적으로 반환
def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    """
    img_dir: 이미지 디렉토리 경로
    img_list: 이미지 파일 리스트
    mask_dir: 마스크 디렉토리 경로
    mask_list: 마스크 파일 리스트
    batch_size: 배치 크기
    """
    L = len(img_list)  # 총 이미지 개수

    # Keras는 제너레이터가 무한히 작동해야 하므로 while True 사용
    while True:
        batch_start = 0 # 시작 인덱스
        batch_end = batch_size # 끝 인덱스

        while batch_start < L:  # 모든 이미지를 처리할 때까지 반복
            limit = min(batch_end, L)  # 배치 끝이 이미지 개수를 초과하지 않도록 제한

            # 배치 크기만큼 이미지와 마스크를 로드
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X, Y)  # 이미지와 마스크의 튜플 반환

            # 다음 배치를 위한 인덱스 업데이트
            batch_start += batch_size   
            batch_end += batch_size