# Project Cell Segmentation

This project was developed as part of the EPFL Machine Learning course with the EPFL Center for Imaging and the help of Mr. Daniel Sage and Mr. Aleix Boquet (2022).

## Authors
- Benjamin Bonnabaud.
- Hugo Robert.
- Yosr Jelassi.

## Summary
This repository contains code used for building a segmentation model of cell images. The task was performed on two different datasets of cells with three different approaches : A basic threshold to have a baseline, a pipeline with a bench filter followed by a random forest classifier, and then a deep learning approach with a U-Net architecture. The data can be found [here](http://celltrackingchallenge.net/2d-datasets/).

## File structure 
```
.
├── README.md
├── data
│   ├── Fluo-N2DHL-Hela
│   │       ├── IMG_TEST
│   │       ├── IMG_TRAIN
│   │       ├── TARGET_TEST
│   │       └── TARGET_TRAIN
│   ├── PhC-C2DH-U373
│   │       ├── IMG_TEST
│   │       ├── IMG_TRAIN
│   │       ├── TARGET_TEST
│   │       └── TARGET_TRAIN
│   ├── merge_dataset.py
│   └── split_dataset.py
├── output
│   ├── Custom_RF
│   ├── Feature_importance
│   ├── Naive_RF
│   ├── Threshold
│   └── UNet
├── UNet
│   ├── unet
│   │       ├── __init__.py
│   │       ├── unet_model.py
│   │       └── unet_parts.py
│   ├── utils
│   │       ├── __init__.py
│   │       ├── data_loading.py
│   │       ├── dice_score.py
│   │       ├── IoU.py
│   │       └── utils.py
│   ├── evaluate.py
│   ├── predict.py
│   ├── README.md
│   ├── requirements.txt
│   └── train.py
├── bench_filter.py
├── filters.py
├── tools.py
├── custom_random_forest_cropped.ipynb
├── custom_random_forest.ipynb
├── naive_random_forest_cropped.ipynb
├── naive_random_forest.ipynb
├── threshold.ipynb
└── UNet.ipynb

```

### File description

- `data` folder contain images and targets splitted between train and test to be able to evaluate each model on same images.
- `output` folder contain outputs of each model.
- `UNet` folder contain the code of the UNet. The initial repository is available [here](https://github.com/milesial/Pytorch-UNet). A README is available inside to explain how to run it.
- `bench_filter.py` contain the Bench Filter object coded as a model object from sklearn (with fit and predict method).
- `filters.py` contain some function to filter images.
- `tools.py` contain some useful functions for the analysis pipeline.
- `custom_random_forest_cropped.ipynb` is the notebook for the random forest model with crop images and custom bench filter.
- `custom_random_forest.ipynb` is the notebook for the random forest model with custom bench filter.
- `naive_random_forest_cropped.ipynb` is the notebook for the random forest model with crop images and naive bench filter.
- `naive_random_forest.ipynb` is the notebook for the random forest model with naive bench filter.
- `threshold.ipynb` is the notebook for the threshold model.
- `UNet.ipynb` is the notebook for the UNet model.


## Requirements
- Python 3
  - `numpy`
  - `pandas`
  - `sklearn`
  - `skimage`
  - `scipy`
  - `tqdm`
  - `torch`
  - `torchvision`
  - `torchmetrics`
  - `wandb`
  - `matplotlib`

  
## Usage

Place the data in the `data` folder. The data, can be downloaded [here](http://celltrackingchallenge.net/2d-datasets/). Then, successively run merge_dataset.py and split_dataset.py. It will move each images in the right folder.

In order to generate our final submission file, you have to run : 

```

```


## Results
Our best model is UNet, but trained with GPUs. This point is clarified in the report. It yielded a mean classification IoU of 94% on both datasets.