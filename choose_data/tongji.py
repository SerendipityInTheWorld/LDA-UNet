import json

def count_splits_with_data_mask(json_file, output_txt=None):
    """
    统计每一折的 train、val、test 数据集中的 data 和 mask 数量。
    
    Args:
        json_file (str): 包含五折交叉验证数据划分的 JSON 文件路径。
        output_txt (str): 可选，输出统计结果的文本文件路径。
    """
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 初始化统计结果
    stats = {}

    # 遍历每一折
    for fold_name, fold_data in data.items():
        train_data_count = len(fold_data.get("train", {}).get("data", []))
        train_mask_count = len(fold_data.get("train", {}).get("mask", []))
        
        val_data_count = len(fold_data.get("val", {}).get("data", []))
        val_mask_count = len(fold_data.get("val", {}).get("mask", []))
        
        test_data_count = len(fold_data.get("test", {}).get("data", []))
        test_mask_count = len(fold_data.get("test", {}).get("mask", []))
        
        # 保存统计结果
        stats[fold_name] = {
            "train": {"data": train_data_count, "mask": train_mask_count},
            "val": {"data": val_data_count, "mask": val_mask_count},
            "test": {"data": test_data_count, "mask": test_mask_count}
        }

    # 打印统计结果
    print("Statistics of each fold:")
    for fold_name, counts in stats.items():
        print(f"{fold_name}:")
        print(f"  Train: Data={counts['train']['data']}, Mask={counts['train']['mask']}")
        print(f"  Val: Data={counts['val']['data']}, Mask={counts['val']['mask']}")
        print(f"  Test: Data={counts['test']['data']}, Mask={counts['test']['mask']}")

    # 如果指定了输出文件，则写入文件
    if output_txt:
        with open(output_txt, 'w') as f:
            f.write("Statistics of each fold:\n")
            for fold_name, counts in stats.items():
                f.write(f"{fold_name}:\n")
                f.write(f"  Train: Data={counts['train']['data']}, Mask={counts['train']['mask']}\n")
                f.write(f"  Val: Data={counts['val']['data']}, Mask={counts['val']['mask']}\n")
                f.write(f"  Test: Data={counts['test']['data']}, Mask={counts['test']['mask']}\n")
        print(f"Statistics saved to {output_txt}")

# 主函数
if __name__ == "__main__":
    # 输入 JSON 文件路径
    json_file = "kfold_splits.json"
    
    # 输出统计结果的文本文件路径（可选）
    output_txt = ""
    
    # 执行统计
    count_splits_with_data_mask(json_file, output_txt)
    print("统计完成！")