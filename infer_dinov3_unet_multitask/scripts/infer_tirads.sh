#!/bin/bash
# =============================================================
# TIRADS 五分类推理示例（含标签文件，输出 CSV + 指标日志）
# =============================================================

# ---------------------- 配置 ----------------------
IMAGE_DIR="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/images/"
CHECKPOINT="/mnt/wangbd8/workspace/ThyroidAgent/dino_unet_multitask/checkpoints/TIRADS/train_TIRADS_Cine-Clip/dino_unet_train_TIRADS_Cine-Clip_epoch_50.pth"
LABEL_FILE="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Cine-Clip/test/Cine-Clip_test_label.json"
LABEL_FIELD="tirads"

OUTPUT="./results/tirads_preds.csv"
LOG_FILE="./results/tirads_metrics.log"

CUDA_DEVICE=0
BATCH_SIZE=16
IMG_SIZE=224
USE_DILATION="False"

# Bootstrap 参数
N_BOOT=2000
CI=0.95
SEED=0

# ---------------------- 执行 ----------------------
python infer_classification.py \
    --image_dir "$IMAGE_DIR" \
    --checkpoint "$CHECKPOINT" \
    --num_classes 5 \
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
    --seed $SEED
