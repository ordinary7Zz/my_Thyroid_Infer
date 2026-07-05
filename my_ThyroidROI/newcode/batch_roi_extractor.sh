#!/bin/bash

python batch_roi_extractor.py \
    --mode images \
    --checkpoint /mnt/wangbd8/workspace/ThyroidAgent/ThyroidROI/outputs/best_dice_model.pth \
    --input_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images \
    --output_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/Classifaction_Data/Malignant_ultrasound_images_cropped