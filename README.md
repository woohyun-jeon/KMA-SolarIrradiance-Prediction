# Spatio-temporal forecasting with KMA ASOS dataset
This repository contains the implementation of deep learning based solar irradiance prediction using KMA ASOS dataset.

## Prerequisites
* python >= 3.10
* torch >= 2.2.0
* torchvision >= 0.17.0

## Usage
1) Clone the repository and install the required dependencies with the following command:
```
$ git clone https://github.com/woohyun-jeon/KMA-SolarIrradiance-Prediction.git
$ cd KMA-SolarIrradiance-Prediction
$ pip install -r requirements.txt
```
2) Download datasets from here:
https://drive.google.com/drive/folders/1zgrBxALwCSSFsa7-RHt0UB4d7Gz7pPjy?usp=sharing

* It is important to mention that "data_path" argument in "configs.yaml" file should be properly adjusted.
* Plus, "save_dir" and "model_dir" argument, indicating output directory of prediction and model files, should be properly adjusted.

3) Run main.py code with the following command:
```
$ cd src
$ python main.py
```