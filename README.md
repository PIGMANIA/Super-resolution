<a href="https://github.com/HeaseoChung/DL-Optimization/tree/master/Python/TensorRT/x86"><img src="https://img.shields.io/badge/-Documentation-brightgreen"/></a>

# AutomaticSR
- This repository helps to train and test various deep learning based super-resolution models by changing few configurations(.yaml)

## Contents
- [News](#updates)
- [Models](#models)
- [Features](#features)
- [Usage](#usage)
- [Citation](#citation)

## Updates
- **_News (2022-07-23)_**: Implemented multi-gpus train approach using pytorch distributed data parallel
- **_News (2022-07-19)_**: Implemented upsampler module at end of Scunet model
- **_News (2022-07-05)_**: Implemented test codes for video and image
- **_News (2022-07-02)_**: Implemented train codes for super-resolution models such as EDSR, BSRGAN, RealESRGAN, SwinIR and SCUnet

## Models
- BSRGAN
- EDSR
- Real-ESRGAN
- SCUNET
- SwinIR

## Features
- Automated in training super-resolution using PSNR and GAN processes
- Automated in testing super-resolution models for image and video

## Usage

### 1. The tree shows config directories in the repository
```
configs/
├── models
│   ├── BSRGAN.yaml
│   ├── EDSR.yaml
│   ├── RealESRGAN.yaml
│   └── SCUNET.yaml
│   └── SwinIR.yaml
├── test
│   ├── image.yaml
│   └── video.yaml
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
    dir: ./outputs/train/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - train: train_type # PSNR, GAN
  - models: model_name # EDSR
```

```yaml
### PSNR or GAN.yaml

### A user should specify train directory in datasets
dataset:
  train_dir: datasets_path # /datasets/train
```

```yaml
### model.yaml
### A user should specify the path of checkpoint in order to resume your train
generator:
  path: "" # /model_zoo/edsr.pth

discriminator:
  path: "" # /model_zoo/discriminator.pth
```

#### Test

```yaml
### test.yaml 
### A user should change model_name and test_type to test various models

hydra:
  run:
    dir: ./outputs/test/${now:%Y-%m-%d}/${now:%H-%M-%S}

defaults:
  - _self_
  - test: test_type # image, video
  - models: model_name # EDSR
```

```yaml
### model.yaml
### A user should specify the path of pre-trained to load weights, in order to inference your model
generator:
  path: "" # /model_zoo/edsr.pth
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