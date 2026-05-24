#!/bin/bash

# 一键运行5折交叉验证，每个模型分别跑完所有fold后再跑下一个模型

MODELS=("LDA_UNet")

# 对于每个模型类型，分别完成5折交叉验证
for MODEL_TYPE in "${MODELS[@]}"; do
    echo "=== Starting training for model: $MODEL_TYPE ==="
    for FOLD in {1..5}; do
    # for FOLD in {5..5}; do
        echo "=== Running Fold $FOLD/5 with Model: $MODEL_TYPE ==="
        python model_train.py --fold $FOLD --model $MODEL_TYPE
    done
    echo "All folds done for model: $MODEL_TYPE!"
done

echo "All models and folds done!"