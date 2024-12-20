import os
import numpy as np
from custom_datagen import imageLoader  # 이미지와 마스크 데이터를 배치(batch) 단위로 반복적으로 반환
import tensorflow as tf
# 모델 생성 클래스인 Model을 가져오는 작업
from tensorflow.keras.models import Model  
# 딥러닝 모델을 구성할 때 사용되는 레이어인 Input과 Dense를 가져오는 작업
from tensorflow.keras.layers import Input, Dense 
#  손실 함수 중 하나인 CategoricalCrossentropy를 가져오는 작업
from tensorflow.keras.losses import CategoricalCrossentropy
# Adam 옵티마이저(Optimizer)를 가져오는 작업
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import glob
import random

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

# 배치 크기 설정
batch_size = 2

# 학습 및 검증 데이터 제너레이터 생성
train_img_datagen = imageLoader(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)
val_img_datagen = imageLoader(val_img_dir, val_img_list, 
                              val_mask_dir, val_mask_list, batch_size)

###########################################################################
# 학습에 사용할 손실 함수, 평가지표, 최적화 함수 정의
wt0, wt1, wt2, wt3 = 0.25, 0.25, 0.25, 0.25     
import segmentation_models_3D as sm

dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))  # Dice 손실 함수
focal_loss = sm.losses.CategoricalFocalLoss()  # Focal 손실 함수
total_loss = dice_loss + (1 * focal_loss)  # 전체 손실 = Dice + Focal

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]  # 정확도 및 IoU 지표
LR = 0.0001  # 학습률 설정
optim = Adam(learning_rate=LR)  # Adam 최적화 함수 사용

#######################################################################
# 모델 생성 및 학습
from unet import unet_model

# 3D U-Net 모델 생성
model = unet_model(IMG_HEIGHT=128, 
                        IMG_WIDTH=128, 
                        IMG_DEPTH=128, 
                        IMG_CHANNELS=3, 
                        num_classes=4)

# 모델 컴파일
model.compile(optimizer=optim, loss=total_loss, metrics=metrics)
print(model.summary())

# 학습
steps_per_epoch = len(train_img_list) // batch_size # 한 에포크(epoch) 동안 실행될 학습 단계 수.
val_steps_per_epoch = len(val_img_list) // batch_size # 검증 단계 수(검증 데이터 개수를 배치 크기로 나눔).

history = model.fit(train_img_datagen, # 학습 데이터 제너레이터.
                    steps_per_epoch=steps_per_epoch, # 에포크당 학습 단계 수.
                    epochs=100, # 학습 반복 횟수
                    verbose=1, # 학습 진행 상황을 출력.
                    validation_data=val_img_datagen, # 검증 데이터 제너레이터.
                    validation_steps=val_steps_per_epoch) # 에포크당 검증 단계 수.

# 모델 저장
model.save('brats_3d.hdf5')

##################################################################
# 학습 및 검증 손실과 정확도 시각화
loss = history.history['loss'] # 각 에포크의 학습 손실 값.
val_loss = history.history['val_loss'] # 각 에포크의 검증 손실 값.
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
'''
# 다시 학습
from keras.models import load_model
# 저장된 모델 불러오기
my_model = load_model(PATH +'sbrats_3d.hdf5')

# 저장된 모델을 불러올 때, 커스텀 손실 함수를 제공
my_model = load_model(PATH +'sbrats_3d.hdf5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss})

# 저장된 모델 불러올 때 IoU 점수 메트릭 추가
my_model = load_model(PATH +'sbrats_3d.hdf5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)})

# 모델 학습을 재개
history2 = my_model.fit(train_img_datagen,         # 학습 데이터 제너레이터
                        steps_per_epoch=steps_per_epoch,  # 에포크 당 스텝 수
                        epochs=1,                 # 학습 에포크 수
                        verbose=1,                # 학습 진행 상황 출력
                        validation_data=val_img_datagen,  # 검증 데이터 제너레이터
                        validation_steps=val_steps_per_epoch)  # 검증 데이터 스텝 수

'''
#################################################
