#!/bin/bash
# =============================================================
# 二分类推理示例（含标签文件，输出 CSV + 指标日志）
# -------------------------------------------------------------
# 阈值选择策略（二选一，均通过 --threshold 指定则优先）：
#   1) 在验证集上计算 Youden 最优阈值（推荐，需提供 VAL_* 变量）
#   2) 不提供验证集 → 使用默认 0.5
# 如需手动指定阈值，设置 THRESHOLD 变量即可（优先级最高）
# =============================================================

# ---------------------- 配置 ----------------------
IMAGE_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/images"
CHECKPOINT="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/checkpoints/train_BM/gamtl_train_multitask_dataset_3/dino_unet_gamtl_train_multitask_dataset_3_epoch_50.pth"
LABEL_FILE="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test/TN3K_test_label.json"
LABEL_FIELD="malignancy"

# 验证集（用于 Youden 阈值计算，留空则使用默认 0.5）
VAL_IMAGE_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/val/images"
VAL_LABEL_FILE="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/val/TN3K_val_label.json"

# 手动指定阈值（设为非空则优先，跳过 Youden；留空则用 Youden 或默认 0.5）
THRESHOLD=""

OUTPUT="./results/binary_preds.csv"
LOG_FILE="./results/binary_metrics.log"

CUDA_DEVICE=0
BATCH_SIZE=16
IMG_SIZE=224
USE_DILATION="False"

# Bootstrap 参数
N_BOOT=2000
CI=0.95
SEED=0

# ---------------------- 组装阈值参数 ----------------------
THRESHOLD_ARG=""
if [ -n "$THRESHOLD" ]; then
    THRESHOLD_ARG="--threshold $THRESHOLD"
elif [ -n "$VAL_IMAGE_DIR" ] && [ -n "$VAL_LABEL_FILE" ]; then
    THRESHOLD_ARG="--val_image_dir $VAL_IMAGE_DIR --val_label_file $VAL_LABEL_FILE"
fi

# ---------------------- 执行 ----------------------
python infer_classification.py \
    --image_dir "$IMAGE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --num_classes 2 \
    --output "$OUTPUT" \
    --label_file "$LABEL_FILE" \
    --label_field "$LABEL_FIELD" \
    --log_file "$LOG_FILE" \
    --img_size $IMG_SIZE \
    --dino_pretrained "False" \
    --use_dilation "$USE_DILATION" \
    --cuda_device $CUDA_DEVICE \
    --batch_size $BATCH_SIZE \
    --n_boot $N_BOOT \
    --ci $CI \
    --seed $SEED \
    $THRESHOLD_ARG
