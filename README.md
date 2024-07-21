# GMoD

## Introduction
This is the implementation of [GMoD: Graph-driven Momentum Distillation
Framework with Active Perception of Disease
Severity for Radiology Report Generation] at MICCAI2024.
![overview]()

## Getting Started
### Requirements

- `einops==0.8.0`
- `matplotlib==3.7.1`
- `nltk==3.8.1`
- `numpy==1.24.2`
- `opencv_python==4.7.0.72`
- `pandas==1.5.3`
- `Pillow==9.4.0`
- `Pillow==10.3.0`
- `scikit_learn==1.2.2`
- `scipy==1.9.1`
- `timm==0.4.12`
- `torch==2.0.0+cu118`
- `torch_geometric==2.3.1`
- `tqdm==4.65.0`



### Download GMoD
You can download the models we trained for each dataset from [here](https://github.com/xzp9999/GMoD-mian/blob/main/data/r2gen.md).

### Datasets
We use two datasets (IU X-Ray and MIMIC-CXR) in our paper.

For `IU X-Ray`, you can download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view?usp=sharing) and then put the files in `data/iu_xray`.

For `MIMIC-CXR`, you can download the dataset from [here](https://physionet.org/content/mimic-cxr/2.0.0/) and then put the files in `data/mimic_cxr`.

NOTE: The `IU X-Ray` dataset is of small size, and thus the variance of the results is large.
There have been some works using `MIMIC-CXR` only and treating the whole `IU X-Ray` dataset as an extra test set.

After downloading the raw dataset, you need to add count_nounphrase.json and mimic-cxr-2.0.0-chexpert.csv to the . /mimic_cxr/ or . /iu_xray/ directory

### Train

Run `bash main_train.py` to train the model.


