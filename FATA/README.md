# Feature Augmentation based Test-time Adaptation (FATA)

기존의 테스트 시점 적응 (Test-Time Adaptation, TTA) 방법들이 적응에 사용할 수 있는 데이터가 제한되는 문제를 해결하기 위해, Feature Augmentation based Test-time Adaptation (FATA) 방법을 제안하였다. 
FATA는 데이터 샘플이 적은 상황에서도 모델의 일반화 성능을 유지하고, 도메인 변화에 효율적으로 적응할 수 있도록 설계되었다.

## Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors

This implementation is build on [Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors 🔗](https://openreview.net/forum?id=9w3iw8wDuE) 
by Jonghyun Lee, Dahuin Jung, Saehyung Lee, Junsung Park, Juhyeon Shin, Uiwon Hwang and Sungroh Yoon (**ICLR 2024 Spotlight, Top-5% of the submissions**).

## Environments  

You should modify [username] and [env_name] in environment.yaml, then  
> $ conda env create --file environment.yaml  

