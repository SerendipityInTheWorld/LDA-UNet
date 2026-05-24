import os, numpy as np, torch
import json
from monai.data import Dataset, DataLoader
import numpy as np
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    RandRotated,
    RandFlipd,
    RandZoomd,
    RandAffined,
    Rand2DElasticd,
    RandGaussianNoised,
    RandAdjustContrastd,
    ScaleIntensityd,
    EnsureTyped,
    ToTensord,
    Compose,
)

def print_data(data):
    print(data.shape)
    print("Unique values in targets:", np.unique(data))  # 应为 0/1/2
def get_transforms(args):
    """返回训练、验证、测试的数据变换管道"""
    if args.mode == 'train':
        train_transforms = Compose([
            # 1. 加载数据并确保通道优先
            LoadImaged(keys=['image', 'mask'], reader='PILReader'),
            EnsureChannelFirstd(keys=['image', 'mask']),

            # 2. 空间变换（几何增强）
            RandRotated(keys=['image', 'mask'], range_x=np.pi/6, prob=0.5),
            RandFlipd(keys=['image', 'mask'], spatial_axis=[0, 1], prob=0.5),
            RandZoomd(
                keys=['image', 'mask'],
                mode=['bilinear', 'nearest'],
                min_zoom=0.8,
                max_zoom=1.2,
                prob=0.5,
                align_corners=[True, None]
            ),
            RandAffined(
                keys=['image', 'mask'],
                mode=('bilinear', 'nearest'),
                prob=0.5,
                rotate_range=np.pi/6,
                translate_range=(10, 10),
                scale_range=(0.1, 0.1)
            ),
            Rand2DElasticd(
                keys=['image', 'mask'],
                mode=('bilinear', 'nearest'),
                prob=0.3,
                spacing=(10, 10),  # 替代原来的sigma_range
                magnitude_range=(50, 100),
                spatial_size=(256, 256)  # 需要指定输出尺寸
            ),

            # 3. 强度变换
            RandGaussianNoised(keys=['image'], prob=0.2, mean=0, std=0.05),
            RandAdjustContrastd(keys=['image'], prob=0.5, gamma=(0.8, 1.2)),

            # 4. 标准化
            ScaleIntensityd(keys=['image'], minv=0, maxv=1),
            EnsureTyped(keys=['image', 'mask'], dtype=('float32', 'int64')),
            ToTensord(keys=['image', 'mask']),
        ])

        val_transforms = Compose([
            LoadImaged(keys=['image', 'mask'], reader='PILReader'),
            EnsureChannelFirstd(keys=['image', 'mask']),
            ScaleIntensityd(keys=['image'], minv=0, maxv=1),
            EnsureTyped(keys=['image', 'mask'], dtype=('float32', 'int64')),
            ToTensord(keys=['image', 'mask']),
        ])
        return train_transforms, val_transforms

    elif args.mode == 'test':
        test_transforms = Compose([
            LoadImaged(keys=['image', 'mask'], reader='PILReader'),
            EnsureChannelFirstd(keys=['image', 'mask']),
            ScaleIntensityd(keys=['image'], minv=0, maxv=1),
            EnsureTyped(keys=['image', 'mask'], dtype=('float32', 'int64')),
            ToTensord(keys=['image', 'mask']),
        ])
        return test_transforms


def get_file(args):
    if args.mode == 'train':
        with open(args.kfold_file) as f:
            splits = json.load(f)

        fold_data = splits[f'fold_{args.fold}']
        train_items = [{'image': img, 'mask': mask} 
                    for img, mask in zip(fold_data['train']['data'], 
                                        fold_data['train']['mask'])]
        val_items = [{'image': img, 'mask': mask} 
                for img, mask in zip(fold_data['val']['data'], 
                                    fold_data['val']['mask'])]
        return train_items, val_items
    elif args.mode == 'test':
        with open(args.kfold_file) as f:
            splits = json.load(f)
        fold_data = splits[f'fold_{args.fold}']
        test_items = [{'image': img, 'mask': mask} 
                for img, mask in zip(fold_data['test']['data'], 
                                    fold_data['test']['mask'])]
        # 对 test_items 进行排序（按 image 文件名排序）
        test_items.sort(key=lambda x: os.path.basename(x['image']))
        # 从 test_items 中提取 mask 文件名（去掉扩展名）
        mask_names = [os.path.splitext(os.path.basename(item['mask']))[0] for item in test_items]

        return test_items,mask_names
    