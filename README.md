# 뇌 종양 이미지 세그멘테이션과 의료 데이터베이스를 활용한 <br/>통합 진단 생태계 구축
***2020272037 의공학부 김원진***

>Project idea 12. Medical Image Segmentation
>-------------------------------------------
> 
>Objective:
>
>Develop a model to segment and classify regions in medical images (e.g., MRI, CT scans).
>
>Approach:
>
>Use a medical image dataset (e.g., BraTS for brain tumor segmentation).
>
>Preprocess the images (e.g., normalization, resizing).
>Implement a U-Net or a similar architecture for image segmentation.
>
>Train the model on labeled images to classify and segment regions of interest.
>
>Evaluate segmentation performance using Dice coefficient, IoU, and visual inspection.

## 프로젝트 설명

![제목 없는 다이어그램 drawio (3)](https://github.com/user-attachments/assets/901f884e-2dcb-4017-b9b3-6f407b783b08)

### 목적

이 프로젝트의 목표는 U-Net 기반 이미지 세그멘테이션 기술을 활용하여 뇌 종양 부위를 자동으로 세그멘테이션하고, 이를 시각적으로 확인할 수 있도록 하는 것입니다.

또한, 환자가 진료 전에 작성한 증상 데이터를 통해 관련 정보를 포함한 데이터베이스와 연계하여 의심되는 뇌 손상 부위를 예측하고, 학습된 모델과의 교차 검증(Cross-check)을 통해 최종적으로 종양 부위를 확정합니다.

이러한 과정을 수행하는 이유는, 뇌 종양의 특성상 같은 위치의 종양이라도 환자마다 다른 증상과 장애를 보일 수 있어, 단순히 이미지 세그멘테이션 결과만으로는 정확한 진단이 어렵기 때문입니다.

마지막으로, 확정된 종양 부위를 기반으로 병원 정보 데이터베이스를 활용하여 최적의 병원과 의사 정보를 제공함으로써, 통합적인 진단 생태계를 구축하는 것을 최종 목표로 합니다.


## 프로젝트 필수 개념

< 더보기 클릭 🖱️ >

<details>
<summary>U-Net</summary>

#### 1. U-Net
![image](https://github.com/user-attachments/assets/2b3a9308-19ed-48ca-9c7f-ad903a5ea0f5)

<details>
<summary>코드 보기</summary>
  
``` python
def unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    """
    3D U-Net 모델을 생성합니다. 

    매개변수:
    - IMG_HEIGHT: 입력 볼륨의 높이
    - IMG_WIDTH: 입력 볼륨의 너비
    - IMG_DEPTH: 입력 볼륨의 깊이(슬라이스 수)
    - IMG_CHANNELS: 입력 데이터의 채널 수 (예: 3채널이면 Flair, T1ce, T2)
    - num_classes: 출력 세그멘테이션 클래스 수

    반환값:
    - model: 3D U-Net 모델 객체
    """

    # 입력 레이어
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    # **수축 경로(Contraction Path, Encoder)**
    # 블록 1
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    c1 = Dropout(0.1)(c1)  # 과적합 방지를 위한 드롭아웃
    c1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c1)
    p1 = MaxPooling3D((2, 2, 2))(c1)  # 다운샘플링(해상도 절반 감소)

    # 블록 2
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c2)
    p2 = MaxPooling3D((2, 2, 2))(c2)

    # 블록 3
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c3)
    p3 = MaxPooling3D((2, 2, 2))(c3)

    # 블록 4
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c4)
    p4 = MaxPooling3D(pool_size=(2, 2, 2))(c4)

    # 병목 지점 (가장 깊은 레이어)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv3D(256, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c5)

    # **확장 경로(Expansive Path, Decoder)**
    # 블록 6
    u6 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(c5)  # 업샘플링
    u6 = concatenate([u6, c4])  # 스킵 연결(Skip Connection)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c6)

    # 블록 7
    u7 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c7)

    # 블록 8
    u8 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c8)

    # 블록 9
    u9 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(c9)

    # 출력 레이어
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(c9)

    # 모델 생성
    model = Model(inputs=[inputs], outputs=[outputs])
    # 모델 구조 출력
    model.summary()

    return model
```
</details>

U-Net은 Biomedical 분야에서 이미지 분할(Image Segmentation)을 목적으로 제안된 End-to-End 방식의 Fully-Convolutional Network 기반 모델입니다. 

모델은 수축 단계(Contracting Path)와 팽창 단계(Expanding Path)로 구성되어 있으며, 이는 Coarse Map에서 Dense Prediction을 얻기 위해 설계되었습니다. 

또한, U-Net은 FCN(Fully Convolutional Network)의 Skip Architecture 개념을 활용하여 얕은 층의 특징맵을 깊은 층의 특징맵과 결합하는 방식을 제안했습니다. 

이러한 Feature Hierarchy 결합을 통해, Segmentation이 요구하는 Localization(위치 정보)과 Context(의미 정보) 간의 트레이드오프를 효과적으로 해결할 수 있습니다.

***Contracting Path (수축 단계)***  
- 3×3 Convolution을 두 차례 반복 (패딩 없음)  
- 활성화 함수: ReLU  
- 2×2 Max-Pooling (stride: 2)  
- Down-sampling마다 채널 수를 2배로 증가  

***Expanding Path (팽창 단계)***  
- 2×2 Convolution ("Up-convolution")  
- 3×3 Convolution을 두 차례 반복 (패딩 없음)  
- Up-convolution마다 채널 수를 절반으로 감소  
- 활성화 함수: ReLU  
- Up-convolution된 특징맵을 Contracting Path의 Cropped된 특징맵과 연결(Concatenation)  
- 마지막 레이어에서 1×1 Convolution 수행  

위와 같은 구조로, U-Net은 총 23개의 Fully Convolutional Layers를 갖춘 네트워크로 설계되었습니다. 

이러한 구성은 Localization(위치 정보)와 Context(의미 정보)를 균형 있게 처리하여 높은 성능의 이미지 분할 결과를 제공하기 때문에 이번 프로젝트에서 사용 되었습니다.
</details>

<details>
<summary>Dice 손실 함수</summary>

#### 2. Dice 손실 함수

DICE 손실 함수(Dice Loss)는 주로 의료 영상 분석과 같은 분야에서 세그멘테이션 문제에 많이 사용됩니다. 

이 손실 함수는 이진 분류 작업에서 두 샘플 집합의 유사도를 측정하기 위해 사용되며, 특히 불균형한 데이터셋에서 좋은 성능을 보입니다.

- DICE 계수

  ![image](https://github.com/user-attachments/assets/243c5d91-9e78-4335-9161-d9cddaab58e1)

  ![image](https://github.com/user-attachments/assets/ffbc778b-4799-44d7-8eea-f39da78e14b5)
  
- DICE 손실함수
  
  ![image](https://github.com/user-attachments/assets/47d7ca0c-a710-4203-acfb-21141e9d3298)
  
  DICE 손실 함수는 1에서 DICE 계수를 뺀 값으로 정의됩니다.
  이는 계수가 1에 가까울수록 손실이 작아지며, 예측과 실제 값 사이의 유사도가 높음을 의미합니다.

- 가중 DICE 손실 함수

  ![image](https://github.com/user-attachments/assets/a382ccd5-f486-4d5e-9b85-cb297f0bf087)

  가중 DICE 손실 함수는 class_weights를 통해 클래스별 중요도를 설정할 수 있습니다.

  코드에서는 가중 DICE 손실 함수를 사용했고 모두 동일하게 0.25를 부여했습니다.

   ``` python
  
  import segmentation_models_3D as sm
  dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3]))  
  
  ``` 
</details>

<details>
<summary>Focal 손실 함수</summary>

#### 3. Focal 손실 함수

Focal 손실 함수는 불균형 데이터 문제를 해결하기 위해 설계된 손실 함수로, 어려운 샘플에 더 큰 가중치를 부여하여 학습을 집중시킵니다.

   ``` python
  
  focal_loss = sm.losses.CategoricalFocalLoss()  # Focal 손실 함수
  
  ``` 

</details>

<details>
<summary>전체 손실 함수</summary>

#### 4. 전체 손실 함수

Dice Loss: 분할 정확도를 높이기 위해 예측 마스크와 실제 마스크 간의 겹침 정도를 평가합니다.

Focal Loss: 클래스 불균형 문제를 해결하고 어려운 샘플에 더 집중하도록 설계합니다.

전체 손실 함수: 두 손실을 합산하여 분할 성능(Localization)과 학습 안정성(Class Imbalance 해결)을 동시에 개선했습니다.

   ``` python
  
  total_loss = dice_loss + (1 * focal_loss)  # 전체 손실 = Dice + Focal
  
  ``` 
</details>

<details>
<summary>정확도 및 IoU 지표</summary>

#### 5. 정확도 및 IoU 지표
정확도 (Accuracy): 전체 픽셀 중 올바르게 예측한 비율입니다.

IoU (Intersection over Union): 예측된 영역과 실제 영역의 겹침 비율입니다.


   ``` python
  
  metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]  # 정확도 및 IoU 지표
  
  ```

</details>

## 프로젝트 설치

***- 필요한 라이브러리***
1. TensorFlow
2. segmentation-models-3D
3. psycopg

***- 필요한  데이터셋***

  [BraTS2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

## 프로젝트 사용법 

### 각 파일의 역할

***`get_data.py`*** 

의료 이미지 데이터셋(MICCAI BraTS 2020)을 정규화 하고 유용한 데이터를 필터링합니다.

***`custom_datagen.py`***

이미지 및 마스크 데이터를 로드하고 배치(batch) 단위로 반환합니다.

***`unet.py`***

3D U-Net 모델 구조를 정의합니다.

***`train.py`***

학습 데이터와 검증 데이터를 사용하여 3D U-Net 모델을 학습시킵니다.
  손실 함수 및 평가지표를 설정하고, 학습 결과를 시각화합니다.

***`load.py`***

모델을 로드하여 IoU 계산 및 시각화를 통해 성능을 평가합니다.

***`search.py`***

학습된 모델을 사용하여 이미지 세그멘테이션을 수행하고 데이터베이스에서 관련 정보를 검색합니다. 

환자 정보와 뇌 증상을 기반으로 병원 데이터를 시각화하여 제공합니다.

***`db.py`***

데이터베이스 연결 및 환자, 뇌 증상, 병원 데이터를 조회하는 함수들을 제공합니다.

### 진행 순서

***1. 데이터 준비 ( `get_data.py` )***
  
의료 이미지(.nii)를 로드한 후, `MinMaxScaler`를 사용해 0~1로 정규화합니다.

마스크 데이터는 3D 볼륨으로, 각 픽셀이 특정 클래스(예: 0-배경, 1-활성 종양 등)를 나타내는 라벨링 정보입니다. 

마스크는 정수형으로 변환 후 유효 데이터 비율이 1% 이상일 경우에만 이미지와 마스크를 `.npy` 형식으로 저장합니다.

- 이미지와 마스크 예시

![image](https://github.com/user-attachments/assets/4d667cd7-1098-4cc2-a8be-733d821daac4)


***2. 배치 처리 및 학습***

`imageLoader`를 사용해 배치 단위로 이미지를 로드하고 반환하는 제너레이터를 생성한 후, Dice Loss와 Focal Loss를 조합한 손실 함수를 정의하고 Adam 옵티마이저와 학습률을 설정합니다.  또한, 평가 지표로 accuracy와 IoU를 추가합니다. 

`unet.py`의 `unet_model`을 활용해 3D U-Net 모델을 생성하고 손실 함수, 옵티마이저, 평가 지표로 모델을 컴파일한 뒤, `model.fit()`을 사용해 학습을 진행합니다. 

마지막으로 학습 손실과 정확도를 시각화하고, 학습된 모델을 `brats_3d.hdf5` 파일로 저장합니다.

- 학습 예시 

![320](https://github.com/user-attachments/assets/1c151eae-fe10-4b49-8005-7627f4f3df1c)


***3. 학습 모델 로드 및 평가***

`load.py`는 학습된 모델(`brats_3d.hdf5`)을 불러와 검증 데이터 제너레이터를 생성하고, 모델의 예측 결과와 실제 마스크를 비교하여 IoU를 계산해 성능을 평가합니다. 

***4. 학습된 모델을 활용한 환자 데이터 분석 및 DB 활용***

학습된 3D U-Net 모델을 불러와 원하는 뇌 이미지를 세그먼트하고, 이미지 ID를 기반으로 데이터베이스에서 환자의 정보를 조회합니다. 

이후, 시각화된 세그먼트 결과와 데이터베이스에 저장된 환자의 증상을 참조하여 다른 데이터베이스(EX symptom table)에서 증상과 관련된 뇌 부위를 조회하고 결과를 출력합니다. 

이를 통해 세그먼트된 이미지와 증상에 따른 종양 부위를 크로스 체크하여, 최종적으로 환자의 종양과 관련된 병원 정보를 병원 데이터베이스에서 조회하고 제공합니다.

- 환자 DB 예시

![image](https://github.com/user-attachments/assets/88824798-50a7-42a1-b8d8-8ed3a3049c69)

## 학습 결과
Batch 크기: 2

Epoch 수: 100

Learning Rate: 0.0001

(구글 코랩을 활용)

***1. Training and validation loss***

![다운로드](https://github.com/user-attachments/assets/adb3e252-8c23-4bae-8479-d0ff57284466)

손실 값이 꾸준히 감소하고 과적합 없이 안정적인 학습을 보여주었습니다.

하지만 손실 값은 매우 큰 편입니다..

***2. Training and Validation Accuracy***

![다운로드 (1)](https://github.com/user-attachments/assets/73bc2d2f-20d7-485c-87ec-d2bd8c1fe9d9)

학습 정확도와 검증 정확도가 약 99% 이상을 보여주었습니다.

***3. 시각적 평가 및 IoU***

![Figure_1](https://github.com/user-attachments/assets/7ceb58ca-fe59-4938-8f72-167344e76340)

IoU 0.8177로 높은 정확도로 세그먼트 수행했습니다.
시각적 평가에서도 실제 마스크와 예측 결과가 유사하게 나타났습니다.

### 종합 평가

코랩의 시간적 한계로 인해 손실 값이 다소 큰 편이지만, 전반적으로 안정적이고 좋은 결과를 나타냈습니다.

## 최종 실행화면

***1. 환자의 MRI 이미지*** 

![1](https://github.com/user-attachments/assets/2c6631e4-8537-4bd2-ab49-be55996abc23)

***2. 종양 부위의 세그멘데이션 및 뇌 부위*** 

![2](https://github.com/user-attachments/assets/61c588e3-eec2-42c5-9d1a-65d9aef93777)

***3. 환자 데이터와 증상을 기반으로 한 종양 부위 예측***

![3](https://github.com/user-attachments/assets/046bc788-71a4-402d-983a-e0ee8f42cf47)

***4. 크로스 체크를 통한 최종 종양 부위 확인 및 관련 병원 명단 제공***

![5](https://github.com/user-attachments/assets/d6644679-a612-406d-bf4b-b34c6cc8b1e1)

## 보완점

처음에는 이미지 세그멘테이션을 통해 종양 부위를 좌표로 감지하고 손상 영역을 자동으로 알려주는 시스템을 구현하려 했으나, 기술적 한계로 인해 실패했습니다.

따라서 현재는 사람이 직접 종양 부위를 확인하고 데이터베이스에 입력해야 하는 불편함이 존재합니다.

이에 따라, 차기에는 YOLO나 Faster R-CNN과 같은 기술을 활용하여 이러한 한계를 극복할 계획입니다.

## 참고자료

https://velog.io/@joongwon00/3D-UNET%EC%9D%84-%EC%9D%B4%EC%9A%A9%ED%95%9C-CTMRI-Image-segmentation-3.-%EC%BD%94%EB%93%9C-%EB%8F%8C%EB%A6%AC%EA%B8%B0

https://medium.com/@msmapark2/u-net-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a

https://bruders.tistory.com/77

https://github.com/bnsreenu/python_for_microscopists

