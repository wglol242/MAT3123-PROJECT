import os # 파일 관리
import numpy as np # 행렬 계산
import nibabel as nib # 의료 이미지 데이터 처리(nil)
import glob # 파일 경로 패턴을 처리
import tensorflow as tf
from tensorflow.keras.utils import to_categorical # 원-핫 인코딩
import matplotlib.pyplot as plt # 그래프 라이브러리
from tifffile import imsave # 이미지를 TIFF 파일 형식으로 저장
from sklearn.preprocessing import MinMaxScaler # 데이터를 0과 1 사이로 정규화

# 데이터를 정규화하기 위한 객체를 생성
scaler = MinMaxScaler()
PATH = ''

# 지정된 파일 경로 패턴에 일치하는 모든 파일의 경로를 반환하고 알파벳순으로 정렬.
t2_list = sorted(glob.glob(PATH + 'MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
t1ce_list = sorted(glob.glob(PATH + 'MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
flair_list = sorted(glob.glob(PATH + 'MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
mask_list = sorted(glob.glob(PATH + 'MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

for img in range(len(t2_list)): 
    print("이미지와 마스크 숫자: ", img)
      
    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    #  3D => 2D => 정규화 => 3D (값을 줄임.), MinMaxScaler는 2D만 지원하므로 변환 필요!
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 3  # 마스크 값 4를 3으로 변환 (파일 오류)
    
    # 결합 후 새로운 축 추가
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]  # 크기 조정
    temp_mask = temp_mask[56:184, 56:184, 13:141] # 크기 조정

    #  각 클래스 라벨의 분포(픽셀 수)를 계산
    val, counts = np.unique(temp_mask, return_counts=True)
    
    # 유용한 데이터가 1% 이상인 경우 저장 => 필요없는 데이터 제거
    if (1 - (counts[0] / counts.sum())) > 0.01:  
        print("Save")
        temp_mask = to_categorical(temp_mask, num_classes=4)
        np.save(f'{PATH}/MICCAI_BraTS2020_TrainingData/result/images/image_{img}.npy', temp_combined_images)
        np.save(f'{PATH}/MICCAI_BraTS2020_TrainingData/result/masks/mask_{img}.npy', temp_mask)
    else:
        print("Useless")   
