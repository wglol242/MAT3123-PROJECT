import os
import numpy as np
from custom_datagen import imageLoader  # 이미지와 마스크 데이터를 배치(batch) 단위로 반복적으로 반환
from keras.models import load_model
from matplotlib import pyplot as plt
import random
import psycopg
from psycopg import sql
from db import *

####################################################
# 학습,검증 데이터 및 마스크 디렉토리 설정
PATH = ''
train_img_dir = PATH +"MICCAI_BraTS2020_TrainingData/result/images/"
train_mask_dir = PATH + "MICCAI_BraTS2020_TrainingData/result/masks/"
val_img_dir = train_img_dir
val_mask_dir = train_mask_dir

# 이미지와 마스크 파일 리스트를 만듬.
train_img_list = os.listdir(train_img_dir)
train_mask_list = os.listdir(train_mask_dir)
val_img_list = train_img_list
val_mask_list = train_mask_list

###########################################################################

# 예측을 위해 모델을 불러오기 (컴파일 생략)
my_model = load_model(PATH +'brats_3d.hdf5', 
                      compile=False)

# IoU 확인 (테스트 데이터셋에서 배치 단위로 확인)
# Keras의 내장 IoU 메트릭 사용 (TF > 2.0에서만 작동)
from keras.metrics import MeanIoU

batch_size = 8  # 테스트 배치 크기 설정
test_img_datagen = imageLoader(val_img_dir, val_img_list,  # 검증 데이터 제너레이터 생성
                               val_mask_dir, val_mask_list, batch_size)

# 제너레이터를 통해 배치 데이터 가져오기
test_image_batch, test_mask_batch = test_img_datagen.__next__()

# 마스크 데이터를 클래스 값으로 변환
test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
# 모델 예측 수행
test_pred_batch = my_model.predict(test_image_batch)
# 예측 데이터를 클래스 값으로 변환
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

# IoU 계산
n_classes = 4  # 클래스 수 설정
IOU_keras = MeanIoU(num_classes=n_classes)  # IoU 메트릭 초기화
IOU_keras.update_state(test_pred_batch_argmax, test_mask_batch_argmax)  # IoU 계산
print("Mean IoU =", IOU_keras.result().numpy())  # IoU 결과 출력

#############################################
# 테스트 이미지에 대한 개별 예측 수행
img_num = 54  # 예측에 사용할 테스트 이미지 번호

HOST = ''
DBNAME = ''
USER = ''
PASSWORD = ''

CONNECTION = f"host={HOST} dbname={DBNAME} user={USER} password={PASSWORD}"
patient = patient_search(CONNECTION=CONNECTION, img_num = img_num)
brain = brain_search(CONNECTION=CONNECTION, symptom = patient[0][2])

# 테스트 이미지와 마스크 로드
test_img = np.load(PATH +"MICCAI_BraTS2020_TrainingData/result/images/image_"+str(img_num)+".npy")
test_mask = np.load(PATH +"MICCAI_BraTS2020_TrainingData/result/masks/mask_"+str(img_num)+".npy")
# 마스크를 클래스 값으로 변환
test_mask_argmax = np.argmax(test_mask, axis=3)

# 모델 입력 형식에 맞게 차원 추가
test_img_input = np.expand_dims(test_img, axis=0)
# 모델 예측 수행
test_prediction = my_model.predict(test_img_input)
# 예측 결과를 클래스 값으로 변환
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0, :, :, :]

# 테스트 이미지 예측 결과 시각화
n_slice = test_prediction_argmax.shape[1] // 2
x_center = 125/2  # 기준 x좌표
y_center = 60  # 기준 y좌표

# 테스트 이미지 시각화
plt.figure(figsize=(10, 8))
plt.title("Image.no: "+str(patient[0][0]), weight='bold', fontsize=30)
plt.imshow(test_img[:, :, n_slice, 1], cmap='gray')  # T1CE 채널 시각화
plt.legend()

# 모델 예측 결과 시각화
plt.figure(figsize=(10, 8))
plt.title("Name: "+patient[0][1], weight='bold', fontsize=30)
plt.imshow(test_prediction_argmax[:, :, n_slice])  # 예측 마스크 시각화
plt.axvline(x=x_center, color='red', linestyle='--')  # x축 선
plt.axhline(y=y_center, color='red', linestyle='--')  # y축 선
plt.text(5, 75, 'Frontal lobe', color='black', fontsize=20, weight='bold', rotation=90)
plt.text(65, 110, 'Parietal lobe', color='black', fontsize=20, weight='bold')
plt.text(30, 125, 'Temporal lobe', color='black', fontsize=15, weight='bold')
plt.text(65, 20, 'Parietal lobe', color='black', fontsize=20, weight='bold')
plt.text(30, 5, 'Temporal lobe', color='black', fontsize=15, weight='bold')
plt.text(120, 77, 'Occipital lobe', color='black', fontsize=20, weight='bold', rotation=90)
plt.legend()

import matplotlib.pyplot as plt

# 데이터를 정리
data = [
    ["Name", str(patient[0][1])],
    ["Age", str(patient[0][3])],
    ["Address", patient[0][4]],
    ["Symptom", patient[0][2]],
    ["Related regions", brain[0][0]]
]
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_visible(False)
ax.axis('off')  # 축 제거
table = plt.table(
    cellText=data,  # 데이터 입력
    colWidths=[0.3, 0.5],  # 열 너비 설정
    loc="center",  # 표 위치
    cellLoc="left"  # 텍스트 정렬
)

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1.5, 2)  # 표 크기 조정

plt.show()

patient_data = input("질병 예상 부위를 입력하세요: ")
hospital_data = hospital_search(CONNECTION=CONNECTION, patient_data = patient_data)

if not hospital_data:
    print("병원 데이터가 없거나 None입니다. 데이터를 확인하세요.")
    exit()

for i, data in enumerate(hospital_data):
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_visible(False)
    ax.axis('off')  # 축 제거

    # 표에 사용할 데이터와 열 제목
    table_data = [['Related regions', data[0]],
                  ['Hospital', data[1]],
                  ['Doctor', data[2]],
                  ['Contact', data[3]]]
    # 표 생성
    table = plt.table(
        cellText=table_data,
        colWidths=[0.3, 0.5],  # 열 너비 설정
        loc="center",  # 표 위치
        cellLoc="left"  # 텍스트 정렬
    )
    # 표 스타일 조정
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.5, 2)
    plt.show()