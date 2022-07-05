<a href="https://github.com/HeaseoChung/DL-Optimization/tree/master/Python/TensorRT/x86"><img src="https://img.shields.io/badge/-Documentation-brightgreen"/></a>

# AutomaticSR
- This repository helps to train and test various deep learning based super-resolution models with simply modifying confings.

## Contents
- [Features](#features)
- [Usage](#usage)

## Features
- Automated super-resolution model train and test processes

## Usage

### 1. The tree shows config directory in the repository
```
configs/
├── models
│   ├── BSRGAN.yaml
│   ├── EDSR.yaml
│   ├── RealESRGAN.yaml
│   └── SCUNET.yaml
├── test
│   ├── image_test.yaml
│   └── video_test.yaml
├── train
│   ├── GAN.yaml
│   └── PSNR.yaml
├── test.yaml
└── train.yaml
```


### 2. A user should modify train.yaml or test.yaml in the config directory to Train or Test super-resolution model

#### Train

```yaml
### train.yaml 
### A user should change model_name and train_type to train various models

hydra:
  run:
    dir: ./outputs/${model_name}/train/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - train: ${train_type}
  - models: ${model_name}
```

```yaml
### PSNR or GAN.yaml

### A user should specify train directory in datasets
dataset:
  train_dir: "datasets"
```

```yaml
### model.yaml
### A user should specify the path of checkpoint in order to resume your train
generator:
  path: ""

discriminator:
  path: ""
```

#### Test

```yaml
### test.yaml 
### A user should change model_name and test_type to test various models

hydra:
  run:
    dir: ./outputs/${model_name}/test/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - test: ${test_type}
  - models: ${model_name}
```

```yaml
### model.yaml
### A user should specify the path of pre-trained to load weights, in order to inference your model
generator:
  path: ""
```

### 3. Run

```python
### trainer.py for train the models
CUDA_VISIBLE_DEVICES=0 python trainer.py
```

```python
### tester.py for test the models
CUDA_VISIBLE_DEVICES=0 python tester.py
```

## Citation
If the repository helps your research or work, please consider citing AutomaticSR.<br>
The following is a BibTeX reference. The BibTeX entry requires the `url` LaTeX package.

``` latex
@misc{chung2022AutomaticSR,
  author =       {Heaseo Chung},
  title =        {{AutomaticSR}: Open source for various super-resolution trainer and tester},
  howpublished = {\url{https://github.com/HeaseoChung/Super-resolution}},
  year =         {2022}
}
```