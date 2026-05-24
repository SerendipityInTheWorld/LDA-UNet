import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from PIL import Image,ImageOps


from network import LDA_UNet
from monai.networks.nets import UNet,AttentionUnet
from monai.networks.nets import UNet,AttentionUnet,SegResNet,SwinUNETR,UNETR

from monai.transforms import AsDiscrete, Activations
from monai.data import Dataset
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from monai.metrics import ConfusionMatrixMetric
# 自定义模块
from choose_data.Dataloader import get_transforms, get_file
# from medcam import medcam
import time
parser = argparse.ArgumentParser(description='Segmentation Testing')

# 数据参数
parser.add_argument('--kfold_file', type=str, default='kfold_splits.json',
                    help='Path to kfold splits json')
parser.add_argument('--fold', type=int, default=1,
                    help='Which fold to use (1-5)')

# 模型参数
parser.add_argument('--model', type=str, default='LDA_UNet',
                    help='Model architecture:unet,LDA_UNet')
parser.add_argument('--in_channels', type=int, default=1,
                    help='Input channels')
parser.add_argument('--num_class', type=int, default=3,
                    help='num_class (including background)')

# 测试参数
parser.add_argument('--mode', type=str, default='test',
                    help='Mode: test or val')
parser.add_argument('--batch_size', type=int, default=1,
                    help='Batch size')
parser.add_argument('--model_path', type=str, default='runs_LiTS/fold1_LDA_UNet/best_model.pth',
                    help='Path to trained model checkpoint')

# 其他参数
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of data loading workers')
parser.add_argument('--save_dir', type=str, default='view_imgmask_LiTS/myNet/fold1',
                    help='Directory to save test results')
parser.add_argument('--seed', type=int, default=12,
                    help='Random seed')
parser.add_argument('--camgrad', type=bool, default=False,
                    help='Random seed')

args = parser.parse_args()
torch.manual_seed(args.seed)

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


from thop import profile

def calculate_flops(model, input_size, device='cuda'):
    model.eval()
    dummy_input = torch.rand(*input_size).to(device)
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    flops_g = flops / 1e9
    print(f'{args.model}模型Flops为：{flops_g:.2f}G')


# 准备目录

save_dir = os.path.join(args.save_dir, f"fold{args.fold}")
os.makedirs(save_dir, exist_ok=True)

# 数据加载
test_transforms = get_transforms(args)
test_files,mask_name = get_file(args)

test_dataset = Dataset(test_files, transform=test_transforms)
test_loader = DataLoader(
    test_dataset, 
    batch_size=args.batch_size, 
    shuffle=False, 
    num_workers=args.num_workers, 
    pin_memory=torch.cuda.is_available()
)

device = setup_device()
# 加载模型参数
checkpoint_path = args.model_path
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
# 模型初始化
if args.model == 'LDA_UNet':
    model = LDA_UNet(
        channels=[24, 48, 96,192,384],
        in_channels=args.in_channels,
        out_channels=args.num_class,
        depth=[1,1,1,1]
    ).to(device)

# if args.camgrad:
#     cam_save = os.path.join(save_dir, 'cam2')
#     if not os.path.exists(cam_save):
#         os.makedirs(cam_save)
#     # model = medcam.inject(model, output_dir=cam_save, save_maps=True)
#     model = medcam.inject(
#     model,
#     output_dir=cam_save,  # 保存注意力图的目录
#     backend="gcam",               # 注意力图生成的后端（如 Grad-CAM）
#     # layer=[("outc")], # 指定目标层的名称
#     layer='auto',
#     label=2,
#     save_maps=True                # 是否保存注意力图
# )
# calculate_flops(model, input_size=(1,1,256,256))

# 加载预训练模型
checkpoint = torch.load(args.model_path,map_location=device)
model.load_state_dict(checkpoint)
print(f"Loaded model from {args.model_path}")
True
# 后处理转换
post_pred = AsDiscrete(argmax=True, to_onehot=args.num_class)
post_label = AsDiscrete(to_onehot=args.num_class)

# 初始化指标（设置 reduction="none" 以获取每个类别的指标）
dice_metric = DiceMetric(include_background=False, reduction="none")
hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="none")
# 初始化混淆矩阵指标（用于计算敏感性和特异性）
confusion_matrix_metric = ConfusionMatrixMetric(
    include_background=False,
    metric_name=["sensitivity", "specificity"],
    reduction="none"
)
def test_model():
    model.eval()
    save_mask_dice_source = []
    with torch.no_grad():
        flage = 0
        for batch in tqdm(test_loader, desc="Testing"):
            inputs = batch["image"].to(device)
            targets = batch["mask"].to(device)
            outputs = model(inputs)

            # 后处理：分离批次并转换为 one-hot
            targets_list = decollate_batch(targets)
            targets_convert = [post_label(t) for t in targets_list]
            
            preds_list = decollate_batch(outputs)
            preds_convert = [post_pred(p) for p in preds_list]

            # 计算指标
            dice_test = dice_metric(y_pred=preds_convert, y=targets_convert)
            hd95_metric(y_pred=preds_convert, y=targets_convert)
            confusion_matrix_metric(y_pred=preds_convert, y=targets_convert)  # 添加混淆矩阵指标计算
            
            if np.mean(dice_test.cpu().numpy()) >0.98:
                save_mask_dice_source.append(list(dice_test.cpu().numpy()[0]))
                for i, pred in enumerate(preds_convert):
                    # 将预测结果从one-hot编码转换为标签图
                    pred_label = torch.argmax(pred, dim=0).cpu().numpy()  # (H, W)

                    # 转换为8位灰度图像
                    pred_image = Image.fromarray((pred_label * (255 // (args.num_class - 1))).astype(np.uint8))

                    # 图像处理：水平翻转 + 右旋转90度
                    pred_image = ImageOps.mirror(pred_image)  # 水平翻转
                    pred_image = pred_image.transpose(Image.Transpose.ROTATE_90)  # 右旋转90度（等价于逆时针旋转270度）

                    # 构造保存路径
                    save_name = f"{mask_name[flage]}.png"  # 替换 flage 为实际变量名
                    save_dir_masks = os.path.join(save_dir, "predicted_masks")
                    os.makedirs(save_dir_masks, exist_ok=True)  # 确保保存目录存在
                    save_path = os.path.join(save_dir_masks, save_name)

                    # 保存预测掩码
                    pred_image.save(save_path)
                    print(f"Saved predicted mask: {save_path}")
            flage += 1
    with open('view_imgmask_kits23/myNet/fold5/output.txt','w') as file:
        for d in save_mask_dice_source:
            line = ','.join(str(x) for x in d) + '\n'
            file.write(line)
    # 聚合所有批次的指标（按类别）
    dice_scores_per_class = dice_metric.aggregate().cpu().numpy()
    hd95_scores_per_class = hd95_metric.aggregate().cpu().numpy()
    max_distance = 256  # 使用对角线距离：np.sqrt(512**2 + 512**2)
    hd95_scores_per_class = np.nan_to_num(hd95_scores_per_class, nan=max_distance, posinf=max_distance, neginf=max_distance)

    # 获取敏感性和特异性
    sensitivity_specificity = confusion_matrix_metric.aggregate()
    sensitivity_per_class = sensitivity_specificity[0].cpu().numpy()  # 敏感性
    specificity_per_class = sensitivity_specificity[1].cpu().numpy()  # 特异性

    # 确保返回值是一维数组
    if len(dice_scores_per_class.shape) > 1:
        dice_scores_per_class = np.nanmean(dice_scores_per_class, axis=0)  # 对批次维度求平均
    if len(hd95_scores_per_class.shape) > 1:
        hd95_scores_per_class = np.nanmean(hd95_scores_per_class, axis=0)  # 对批次维度求平均
    if len(sensitivity_per_class.shape) > 1:
        sensitivity_per_class = np.nanmean(sensitivity_per_class, axis=0)  # 对批次维度求平均
    if len(specificity_per_class.shape) > 1:
        specificity_per_class = np.nanmean(specificity_per_class, axis=0)  # 对批次维度求平均

    # 计算平均值
    avg_dice = np.nanmean(dice_scores_per_class)
    avg_hd95 = np.nanmean(hd95_scores_per_class)
    avg_sensitivity = np.nanmean(sensitivity_per_class)
    avg_specificity = np.nanmean(specificity_per_class)

    # 重置指标状态
    dice_metric.reset()
    hd95_metric.reset()
    confusion_matrix_metric.reset()

    return (
        dice_scores_per_class, 
        hd95_scores_per_class, 
        sensitivity_per_class, 
        specificity_per_class, 
        avg_dice, 
        avg_hd95, 
        avg_sensitivity, 
        avg_specificity
    )


if __name__ == "__main__":
    print("\nStarting testing...")
    
    # 测试模型并获取指标
    (
        dice_scores_per_class, 
        hd95_scores_per_class, 
        sensitivity_per_class, 
        specificity_per_class, 
        avg_dice, 
        avg_hd95, 
        avg_sensitivity, 
        avg_specificity
    ) = test_model()

    # 打印每个类别的结果
    print("\nPer-class Results:")
    for class_idx in range(len(dice_scores_per_class)):
        print(f"Class {class_idx + 1}:")
        print(f"  Dice: {float(dice_scores_per_class[class_idx]):.4f}")  # 转换为 float
        print(f"  HD95: {float(hd95_scores_per_class[class_idx]):.4f}")   # 转换为 float
        print(f"  Sensitivity: {float(sensitivity_per_class[class_idx]):.4f}")  # 转换为 float
        print(f"  Specificity: {float(specificity_per_class[class_idx]):.4f}")  # 转换为 float

    # 转换 NumPy float32 为 Python float
    results = {
        "Overall": {
            "Dice": float(avg_dice),  # 显式转换为 Python float
            "HD95": float(avg_hd95),  # 显式转换为 Python float
            "Sensitivity": float(avg_sensitivity),  # 显式转换为 Python float
            "Specificity": float(avg_specificity),  # 显式转换为 Python float
        },
        "PerClass": {
            f"Class_{i+1}": {
                "Dice": float(dice_scores_per_class[i]),  # 显式转换为 Python float
                "HD95": float(hd95_scores_per_class[i]),   # 显式转换为 Python float
                "Sensitivity": float(sensitivity_per_class[i]),  # 显式转换为 Python float
                "Specificity": float(specificity_per_class[i])   # 显式转换为 Python float
            } for i in range(len(dice_scores_per_class))
        },
        "Model": args.model_path,
        "Fold": args.fold
    }

    # 保存结果到 JSON 文件
    result_file = os.path.join(save_dir, f"{args.save_dir}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)

    print("\nTest Results:")
    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average HD95: {avg_hd95:.4f}")
    print(f"Average Sensitivity: {avg_sensitivity:.4f}")
    print(f"Average Specificity: {avg_specificity:.4f}")
    print(f"Results saved to {result_file}")