import os
import json
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict
import torch
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImage, EnsureChannelFirst, ScaleIntensity,
    RandRotate, RandFlip, RandZoom, RandGaussianNoise,
    Lambdad, AsDiscrete, ToTensor
)
def get_train_transform(img_size=(256, 256),num_classes=2):
    return Compose([
        # 加载数据
        LoadImage(image_only=True, reader='PILReader'),  # 使用PIL读取保证兼容性
        EnsureChannelFirst(),  # 添加通道维度
        
        # 图像预处理
        ScaleIntensity(minv=0.0, maxv=1.0),  # 归一化到[0,1]
        
        # 空间增强
        RandRotate(range_x=np.pi/6, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandZoom(min_zoom=0.8, max_zoom=1.2, prob=0.5, keep_size=True),
        
        # 强度增强
        RandGaussianNoise(prob=0.2, std=0.01),
        
        # 标签处理
        # Lambdad(keys='mask', func=lambda x: (x > 0).astype(np.int8)), # 二值化处理
        # AsDiscrete(keys='mask', to_onehot=num_classes),  # One-Hot编码
        
        # 转换为Tensor
        ToTensor(dtype=torch.float32, track_meta=False)
    ])

def get_val_transform():
    return Compose([
        LoadImage(image_only=True, reader='PILReader'),
        EnsureChannelFirst(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        # Lambdad(keys='mask', func=lambda x: (x > 0).astype(np.int8)),
        # AsDiscrete(keys='mask', to_onehot=num_classes),
        ToTensor(dtype=torch.float32, track_meta=False)
    ])
