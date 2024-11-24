"""
@author: Sreenivas Bhattiprolu

3D U-Net을 사용하여 BraTS 2020 데이터셋의 잘라진 이미지 배치를 학습시키는 코드입니다.

이 코드를 실행하기 전에 데이터 준비 및 커스텀 데이터 제너레이터를 정의해야 합니다.
(이 디렉토리의 다른 파일 참고)

이미지 데이터는 128x128x128x3 형식의 `.npy` 파일이어야 하며,
채널 3은 `test_image_flair`, `test_image_t1ce`, `test_image_t2`를 나타냅니다.

마스크 데이터는 128x128x128x4 형식의 `.npy` 파일이어야 하며,
채널 4는 클래스/레이블(0, 1, 2, 3)을 나타냅니다.

컴퓨팅 리소스에 따라 입력 크기를 변경할 수 있습니다.
"""

# 필요한 라이브러리 불러오기
import os
import numpy as np
from custom_datagen import imageLoader  # 사용자 정의 데이터 제너레이터
#import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random

####################################################
# 학습 데이터 및 마스크 디렉토리 설정
train_img_dir = "BraTS2020_TrainingData/input_data_128/train/images/"
train_mask_dir = "BraTS2020_TrainingData/input_data_128/train/masks/"

# 이미지와 마스크 파일 리스트
img_list = os.listdir(train_img_dir)
msk_list = os.listdir(train_mask_dir)

# 총 이미지 개수
num_images = len(os.listdir(train_img_dir))

# 랜덤으로 하나의 이미지와 마스크를 선택하여 시각화
img_num = random.randint(0, num_images - 1)
test_img = np.load(train_img_dir + img_list[img_num])  # 선택한 이미지 로드
test_mask = np.load(train_mask_dir + msk_list[img_num])  # 선택한 마스크 로드
test_mask = np.argmax(test_mask, axis=3)  # 원-핫 인코딩된 마스크를 클래스 값으로 변환

# 랜덤 슬라이스 선택
n_slice = random.randint(0, test_mask.shape[2])
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

#############################################################
# 각 클래스의 분포를 확인하고 클래스별 가중치를 계산 (선택 사항)
# 가중치를 지정하지 않고 0.25로 동일하게 설정할 수도 있음

import pandas as pd
columns = ['0', '1', '2', '3']
df = pd.DataFrame(columns=columns)
train_mask_list = sorted(glob.glob('BraTS2020_TrainingData/input_data_128/train/masks/*.npy'))

# 각 마스크 파일의 클래스 분포 계산
for img in range(len(train_mask_list)):
    print(img)
    temp_image = np.load(train_mask_list[img])
    temp_image = np.argmax(temp_image, axis=3)  # 클래스 값으로 변환
    val, counts = np.unique(temp_image, return_counts=True)
    zipped = zip(columns, counts)
    conts_dict = dict(zipped)
    df = df.append(conts_dict, ignore_index=True)

# 각 클래스의 총 픽셀 수 계산
label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['2'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4

# 클래스별 가중치 계산: n_samples / (n_classes * n_samples_for_class)
wt0 = round((total_labels / (n_classes * label_0)), 2)
wt1 = round((total_labels / (n_classes * label_1)), 2)
wt2 = round((total_labels / (n_classes * label_2)), 2)
wt3 = round((total_labels / (n_classes * label_3)), 2)

##############################################################
# 학습 및 검증용 데이터 제너레이터 정의
val_img_dir = "BraTS2020_TrainingData/input_data_128/val/images/"
val_mask_dir = "BraTS2020_TrainingData/input_data_128/val/masks/"

train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
val_img_list = os.listdir(val_img_dir)
val_mask_list = os.listdir(val_mask_dir)

# 배치 크기 설정
batch_size = 2

# 학습 및 검증 데이터 제너레이터 생성
train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                              val_mask_dir, val_mask_list, batch_size)

# 데이터 제너레이터에서 하나의 배치를 로드하여 확인
img, msk = train_img_datagen.__next__()

# 배치 내 이미지와 마스크 랜덤 선택 및 시각화
img_num = random.randint(0, img.shape[0] - 1)
test_img = img[img_num]
test_mask = msk[img_num]
test_mask = np.argmax(test_mask, axis=3)

n_slice = random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:, :, n_slice, 0], cmap='gray')  # FLAIR 채널
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')  # T1CE 채널
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:, :, n_slice, 2], cmap='gray')  # T2 채널
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:, :, n_slice])  # 마스크
plt.title('Mask')
plt.show()

###########################################################################
# 학습에 사용할 손실 함수, 평가지표, 최적화 함수 정의
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))  # Dice 손실 함수
focal_loss = sm.losses.CategoricalFocalLoss()  # Focal 손실 함수
total_loss = dice_loss + (1 * focal_loss)  # 전체 손실 = Dice + Focal

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]  # 정확도 및 IoU 지표
LR = 0.0001  # 학습률 설정
optim = keras.optimizers.Adam(LR)  # Adam 최적화 함수 사용

#######################################################################
# 모델 생성 및 학습
from simple_3d_unet import simple_unet_model

# 3D U-Net 모델 생성
model = simple_unet_model(IMG_HEIGHT=128, 
                          IMG_WIDTH=128, 
                          IMG_DEPTH=128, 
                          IMG_CHANNELS=3, 
                          num_classes=4)

# 모델 컴파일
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

# 학습
steps_per_epoch = len(train_img_list) // batch_size
val_steps_per_epoch = len(val_img_list) // batch_size

history = model.fit(train_img_datagen,
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,
                    verbose=1,
                    validation_data=val_img_datagen,
                    validation_steps=val_steps_per_epoch)

# 모델 저장
model.save('brats_3d.hdf5')

##################################################################
# 학습 및 검증 손실과 정확도 시각화
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

############################################################
