import os
import json
import numpy as np
from sklearn.model_selection import KFold
from collections import defaultdict

def create_kfold_splits(data_dir, mask_dir, output_file='kfold_splits.json', k=5, seed=12):
    """
    按病例划分的五折交叉验证
    文件名格式: LiTS_<case_id>_slice_<num>.png 和 LiTS_<case_id>_slice_<num>_mask.png
    """
    # 获取所有数据文件并提取病例ID
    data_files = [f for f in os.listdir(data_dir) 
                 if f.endswith('.png') and not f.endswith('_mask.png')]
    
    # 提取病例ID（格式：LiTS_0_slice_xxx.png → 提取"0"）
    case_ids = sorted(list(set(f.split('_')[1] for f in data_files)))
    
    if len(case_ids) < k:
        raise ValueError(f"病例数({len(case_ids)})少于交叉验证折数({k})")

    # 按病例收集所有切片
    case_dict = defaultdict(list)
    for data_file in data_files:
        parts = data_file.split('_')
        case_id = parts[1]  # 提取病例ID
        mask_file = data_file.replace('.png', '_mask.png')
        
        if not os.path.exists(os.path.join(mask_dir, mask_file)):
            print(f"警告：缺少标签文件 {mask_file}，已跳过")
            continue
            
        case_dict[case_id].append((
            os.path.join(data_dir, data_file),
            os.path.join(mask_dir, mask_file)
        ))

    # 五折交叉验证
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    splits = {}
    case_ids = sorted(case_dict.keys())  # 确保病例顺序固定
    
    for fold, (train_val_idx, test_idx) in enumerate(kf.split(case_ids)):
        # 划分训练验证集和测试集
        train_val_cases = [case_ids[i] for i in train_val_idx]
        test_cases = [case_ids[i] for i in test_idx]
        
        # 从训练验证集中划分验证集 (7:1:2比例)
        val_size = max(1, int(0.07 * len(train_val_cases)))  # 10%验证
        val_cases = train_val_cases[-val_size:]
        train_cases = train_val_cases[:-val_size]
        
        # 构建分割字典
        def get_paths(cases):
            data, mask = zip(*[item for c in cases for item in case_dict[c]])
            return {'data': list(data), 'mask': list(mask)}
        
        splits[f'fold_{fold+1}'] = {
            'train': get_paths(train_cases),
            'val': get_paths(val_cases),
            'test': get_paths(test_cases)
        }

    # 保存为JSON文件
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"成功生成五折交叉验证方案，已保存至 {output_file}")
    print(f"病例分布：总病例数 {len(case_ids)}，每折病例数 ~{len(case_ids)//k}")
    return splits

# 使用示例
data_dir = "/home/lulian/dataset/LiTs_result/data"
mask_dir = "/home/lulian/dataset/LiTs_result/mask"
splits = create_kfold_splits(data_dir, mask_dir)