# Feature Augmentation based Test-time Adaptation (FATA)

기존의 테스트 시점 적응 (Test-Time Adaptation, TTA) 방법들이 적응에 사용할 수 있는 데이터가 제한되는 문제를 해결하기 위해, Feature Augmentation based Test-time Adaptation (FATA) 방법을 제안하였다. 
FATA는 데이터 샘플이 적은 상황에서도 모델의 일반화 성능을 유지하고, 도메인 변화에 효율적으로 적응할 수 있도록 설계되었다.

## Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors

This implementation is build on [Entropy is not Enough for Test-Time Adaptation: From the Perspective of Disentangled Factors 🔗](https://openreview.net/forum?id=9w3iw8wDuE) 
by Jonghyun Lee, Dahuin Jung, Saehyung Lee, Junsung Park, Juhyeon Shin, Uiwon Hwang and Sungroh Yoon (**ICLR 2024 Spotlight, Top-5% of the submissions**).

## Environments  

You should modify [username] and [env_name] in environment.yaml, then  
> $ conda env create --file environment.yaml  

## Dataset
You can download ImageNet-C from a link [ImageNet-C 🔗](https://zenodo.org/record/2235448).  

After downloading the dataset, move to the root directory ([data_root]) of datasets.  

If you run on [ColoredMNIST 🔗](https://arxiv.org/abs/1907.02893) or [Waterbirds 🔗](https://arxiv.org/abs/1911.08731), run  
> $ python pretrain_[dataset_name].py --root_dir [data_root] --dset [dataset_name]

Then datasets are automatically downloaded in your [data_root] directory.  
(ColoredMNIST from [torchvision 🔗](https://pytorch.org/vision/stable/index.html) and ./dataset/ColoredMNIST_dataset.py, Waterbirds from [wilds 🔗](https://pypi.org/project/wilds/) package)

Your [data_root] will be as follows:
```bash
data_root
├── ImageNet-C
│   ├── brightness
│   ├── contrast
│   └── ...
├── ColoredMNIST
│   ├── ColoredMNIST_model.pickle
│   ├── MNIST
│   ├── train1.pt
│   ├── train2.pt
│   └── test.pt
├── Waterbirds
│   ├── metadata.csv
│   ├── waterbirds_dataset.h5py
│   ├── waterbirds_pretrained_model.pickle
│   ├── 001. Black_footed_Albatross
│   ├── 002. Laysan_Albatross
└── └── ...
```
If you don't want to pre-train, you can just copy and paste the [dataset_name]_model.pickle from './pretrained/' directory.

## Experiment

You can run most of the experiments in our paper by  
> $ chmod +x exp_deyo.sh  
> $ ./exp_deyo.sh  

If you want to run on the ImageNet-R or VISDA-2021, you should use main_da.py

You should modify ROOT variable as [data_root] in exp_deyo.sh.  
