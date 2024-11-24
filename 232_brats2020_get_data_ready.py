import os 
import numpy as np
import nibabel as nib
import glob
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave
from sklearn.preprocessing import MinMaxScaler

# 데이터 정규화를 위한 Scaler 초기화
scaler = MinMaxScaler()

# 데이터 경로 설정
TRAIN_DATASET_PATH = 'C:/Users/82102/Desktop/task/MICCAI_BraTS2020_TrainingData/'

# 이미지 하나 불러오기 => 넘파이 3차원 이미지 240x240x155 (MRI flair이미지)
test_image_flair = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_355/BraTS20_Training_355_flair.nii').get_fdata()
# 이미지 정규화(중요!!) 3D => 1D => 3D?
test_image_flair = scaler.fit_transform(test_image_flair.reshape(-1, test_image_flair.shape[-1])).reshape(test_image_flair.shape)

# T1 이미지를 NIfTI 파일에서 로드하고 정규화
test_image_t1 = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_355/BraTS20_Training_355_t1.nii').get_fdata()
test_image_t1 = scaler.fit_transform(test_image_t1.reshape(-1, test_image_t1.shape[-1])).reshape(test_image_t1.shape)

# T1CE 이미지를 NIfTI 파일에서 로드하고 정규화
test_image_t1ce = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_355/BraTS20_Training_355_t1ce.nii').get_fdata()
test_image_t1ce = scaler.fit_transform(test_image_t1ce.reshape(-1, test_image_t1ce.shape[-1])).reshape(test_image_t1ce.shape)

# T2 이미지를 NIfTI 파일에서 로드하고 정규화
test_image_t2 = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_355/BraTS20_Training_355_t2.nii').get_fdata()
test_image_t2 = scaler.fit_transform(test_image_t2.reshape(-1, test_image_t2.shape[-1])).reshape(test_image_t2.shape)

# Segmentation 마스크 데이터를 로드하고 uint8로 변환 (이미지에서 특정 영역(뇌종양, 조직 등)을 표시한 3D 라벨 데이터.)
test_mask = nib.load(TRAIN_DATASET_PATH + 'BraTS20_Training_355/BraTS20_Training_355_seg.nii').get_fdata()
# 소숫점으로 받기 때문에 정수로 변환
test_mask = test_mask.astype(np.uint8)

# 마스크에 포함된 고유 값 출력 (0, 1, 2, 4) => 오류
# 마스크 값 4를 3으로 재매핑 (레이블을 0, 1, 2, 3으로 조정)
test_mask[test_mask == 4] = 3

# 랜덤 슬라이스를 선택하여 이미지 시각화
import random
n_slice = random.randint(0, test_mask.shape[2])  # 랜덤 슬라이스 선택(0~mask 3번째 축)

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.imshow(test_image_flair[:, :, n_slice], cmap='gray')
plt.title('Image flair')

plt.subplot(232)
plt.imshow(test_image_t1[:, :, n_slice], cmap='gray')
plt.title('Image t1')

plt.subplot(233)
plt.imshow(test_image_t1ce[:, :, n_slice], cmap='gray')
plt.title('Image t1ce')

plt.subplot(234)
plt.imshow(test_image_t2[:, :, n_slice], cmap='gray')
plt.title('Image t2')

plt.subplot(235)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')

plt.show()

# 여러 채널 데이터를 결합하여 하나의 배열로 만듦
# FLAIR, T1CE, T2 이미지를 결합하여 다중 채널 4D 배열 생성.
combined_x = np.stack([test_image_flair, test_image_t1ce, test_image_t2], axis=3)

# 배열을 64의 배수 크기로 자르기 => 배경이 어둡기 떄문에
combined_x = combined_x[56:184, 56:184, 13:141]  # 128x128x128x4 크기로 자름
test_mask = test_mask[56:184, 56:184, 13:141]  # 마스크도 동일하게 자름

# 반복 (슬라이스)
n_slice = random.randint(0, test_mask.shape[2])

plt.figure(figsize=(12, 8))
plt.subplot(221)
plt.imshow(combined_x[:, :, n_slice, 0], cmap='gray')
plt.title('Image flair')

plt.subplot(222)
plt.imshow(combined_x[:, :, n_slice, 1], cmap='gray')
plt.title('Image t1ce')

plt.subplot(223)
plt.imshow(combined_x[:, :, n_slice, 2], cmap='gray')
plt.title('Image t2')

plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])
plt.title('Mask')

plt.show()

# 결합된 데이터와 마스크를 파일로 저장
imsave('C:/Users/82102/Desktop/task/MICCAI_BraTS2020_TrainingData/result/combined255.tif', combined_x)
np.save('C:/Users/82102/Desktop/task/MICCAI_BraTS2020_TrainingData/result/combined255.npy', combined_x)

# 저장된 데이터를 다시 불러와 확인
my_img = np.load('C:/Users/82102/Desktop/task/MICCAI_BraTS2020_TrainingData/result/combined255.npy')
# 테스트 마스크에 4추가?? 
test_mask = to_categorical(test_mask, num_classes=4)

# 위 과정을 모든 데이터에 실행

# 모든 이미지에 대해 위 과정을 반복 (Flair, T1CE, T2 이미지 결합 및 자르기)
t2_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
t1ce_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
flair_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
mask_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

for img in range(len(t2_list)):  # 모든 이미지에 대해 반복
    print("Now preparing image and masks number: ", img)
      
    temp_image_t2 = nib.load(t2_list[img]).get_fdata()
    temp_image_t2 = scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
   
    temp_image_t1ce = nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce = scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)
   
    temp_image_flair = nib.load(flair_list[img]).get_fdata()
    temp_image_flair = scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
        
    temp_mask = nib.load(mask_list[img]).get_fdata()
    temp_mask = temp_mask.astype(np.uint8)
    temp_mask[temp_mask == 4] = 3  # 마스크 값 4를 3으로 변환
    
    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]  # 크기 조정
    temp_mask = temp_mask[56:184, 56:184, 13:141]

    val, counts = np.unique(temp_mask, return_counts=True)
    
    # 유용한 데이터가 1% 이상인 경우 저장 => 필요없는 데이터 거름
    if (1 - (counts[0] / counts.sum())) > 0.01:  
        print("Save Me")
        temp_mask = to_categorical(temp_mask, num_classes=4)
        np.save(f'BraTS2020_TrainingData/input_data_3channels/images/image_{img}.npy', temp_combined_images)
        np.save(f'BraTS2020_TrainingData/input_data_3channels/masks/mask_{img}.npy', temp_mask)
    else:
        print("I am useless")   

# 데이터를 훈련, 테스트, 검증 세트로 나누기
import splitfolders
input_folder = 'BraTS2020_TrainingData/input_data_3channels/'
output_folder = 'BraTS2020_TrainingData/input_data_128/'
splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.75, .25))  # 75:25 비율로 분할
