#!/bin/bash
INPUT_DIR="ORIGIN"
BASE_OUTPUT_DIR="ORIGIN-MAKE"

# 创建基础输出目录
mkdir -p $BASE_OUTPUT_DIR

# 所有需要使用的方法
METHODS=("time_shift" "jittering" "time_warp" "magnitude_warp" "random_scaling" "noise_injection" "window_slice" "smoothing" "permutation")

# 对每种方法逐一执行
for METHOD in "${METHODS[@]}"; do
    OUTPUT_DIR="$BASE_OUTPUT_DIR/$METHOD"
    mkdir -p $OUTPUT_DIR

    echo "处理方法: $METHOD"
    python trajectory_augmentation.py --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --methods $METHOD --n_augmentations 1 --visualize
done

echo "所有增强处理完成！"