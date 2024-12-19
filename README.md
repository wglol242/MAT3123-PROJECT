# 뇌 종양 이미지 분할 및 데이터베이스를 활용한 증상 연계

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

## 목차

## 프로젝트 설명

![제목 없는 다이어그램 drawio (3)](https://github.com/user-attachments/assets/901f884e-2dcb-4017-b9b3-6f407b783b08)

### 목적

이 프로젝트의 목표는 **U-Net 기반 이미지 분할(Image Segmentation) 기술**을 활용하여 뇌 종양 부위를 자동으로 분할(segmentation)하고, 이를 시각적으로 확인할 수 있는 **마스크를 생성**하는 것입니다.

또한, 환자가 진료 전에 작성한 **증상 데이터**를 증상 관련 정보를 포함한 **데이터베이스(DB)** 와 연계하여, 의심되는 뇌 손상 부위를 탐지하고, 생성한 종양 마스크와 **교차 검증(Cross-check)** 을 통해 최종적으로 종양 부위를 확정합니다.

이러한 과정을 진행하는 이유는 뇌 종양의 특성상, 같은 위치의 종양이라도 환자마다 다른 증상과 장애를 보일 수 있기 때문에 단순히 이미지 분할 결과만으로는 완전한 진단이 어려운 점을 보완하기 위함입니다.

마지막으로, 확정된 종양 부위와 연관된 병원의 정보를 데이터베이스에서 검색하여 제공함으로써, **종합적인 자동 진단 환경**을 구축하는 것을 궁극적인 목표로 합니다.


### 프로젝트 필수 개념

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

이러한 구성은 Localization(위치 정보)와 Context(의미 정보)를 균형 있게 처리하여 높은 성능의 이미지 분할 결과를 제공합니다.

#### 2. Dice 손실 함수

DICE 손실 함수(Dice Loss)는 주로 의료 영상 분석과 같은 분야에서 세그멘테이션 문제에 많이 사용됩니다. 

이 손실 함수는 이진 분류 작업에서 두 샘플 집합의 유사도를 측정하기 위해 사용되며, 특히 불균형한 데이터셋에서 좋은 성능을 보입니다.

- DICE 계수

  ![image](https://github.com/user-attachments/assets/243c5d91-9e78-4335-9161-d9cddaab58e1)
  
- DICE 손실함수
  
  ![image](https://github.com/user-attachments/assets/47d7ca0c-a710-4203-acfb-21141e9d3298)
  
  DICE 손실 함수는 1에서 DICE 계수를 뺀 값으로 정의됩니다. 이는 계수가 1에 가까울수록 손실이 작아지며, 예측과 실제 값 사이의 유사도가 높음을 의미합니다

#### 3. Focal 손실 함수

#### 4. 정확도 및 IoU 지표

## 프로젝트 설치 및 실행법

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## 프로젝트 사용법 

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

## 학습 결과

Explain what these tests test and why

```
Give an example
```

## 프로젝트 결과

Add additional notes about how to deploy this on a live system

## 참고자료

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds
