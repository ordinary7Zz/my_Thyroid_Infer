import os
import sys
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import torch

from roi_extractor import ROIExtractor

def batch_process(input_root, output_root, checkpoint_path):
    """批量处理单独的图像"""
    # Initialize extractor
    print(f"Initializing ROIExtractor with checkpoint: {checkpoint_path}")
    extractor = ROIExtractor(checkpoint_path)

    # Count total images for progress bar
    image_files = []
    for root, _, files in os.walk(input_root):
        # Copy report_text.json if exists
        if "report_text.json" in files:
            src_json = os.path.join(root, "report_text.json")
            rel_path = os.path.relpath(root, input_root)
            dest_dir = os.path.join(output_root, rel_path)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src_json, os.path.join(dest_dir, "report_text.json"))
            print(f"Copied report_text.json for {rel_path}")

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} images to process.")

    success_count = 0
    error_count = 0

    for image_path in tqdm(image_files, desc="Processing Images"):
        try:
            # Calculate relative path to maintain structure
            rel_path = os.path.relpath(image_path, input_root)
            output_path = os.path.join(output_root, rel_path)

            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Extract ROI
            # extract_roi returns RGB float32 [0, 1]
            roi_rgb = extractor.extract_roi(image_path)

            # Convert to BGR uint8 [0, 255] for OpenCV saving
            roi_bgr = (roi_rgb[:, :, ::-1] * 255).astype(np.uint8)

            # Save image
            cv2.imwrite(output_path, roi_bgr)
            success_count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            error_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")


def batch_process_with_masks(images_dir, masks_dir, output_images_dir, output_masks_dir, checkpoint_path):
    """批量处理图像和对应的mask"""
    # Initialize extractor
    print(f"Initializing ROIExtractor with checkpoint: {checkpoint_path}")
    extractor = ROIExtractor(checkpoint_path)

    # Get all image files
    image_files = []
    for root, _, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} images to process.")

    success_count = 0
    error_count = 0
    skipped_count = 0

    for image_path in tqdm(image_files, desc="Processing Image-Mask Pairs"):
        try:
            # Calculate relative path to maintain structure
            rel_path = os.path.relpath(image_path, images_dir)

            # Find corresponding mask
            mask_path = os.path.join(masks_dir, rel_path)

            # Check if mask exists
            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for {rel_path}, skipping.")
                skipped_count += 1
                continue

            # Output paths
            output_image_path = os.path.join(output_images_dir, rel_path)
            output_mask_path = os.path.join(output_masks_dir, rel_path)

            # Create output directories
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)

            # Extract ROI with crop parameters
            roi_rgb, crop_params = extractor.extract_roi_with_crop_params(image_path)

            # Convert to BGR uint8 [0, 255] for OpenCV saving
            roi_bgr = (roi_rgb[:, :, ::-1] * 255).astype(np.uint8)

            # Load and crop mask using the same parameters
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = cv2.imread(mask_path)
                if mask is not None:
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
                else:
                    raise ValueError(f"无法读取mask: {mask_path}")

            # Crop mask using the same coordinates
            x, y, w, h = crop_params['x'], crop_params['y'], crop_params['w'], crop_params['h']
            cropped_mask = mask[y:y+h, x:x+w].copy()

            # Apply the processed mask to keep only the ROI region
            processed_mask = crop_params['mask']
            cropped_processed_mask = processed_mask[y:y+h, x:x+w]
            cropped_mask = np.where(cropped_processed_mask > 0, cropped_mask, 0).astype(np.uint8)

            # Save cropped image and mask
            cv2.imwrite(output_image_path, roi_bgr)
            cv2.imwrite(output_mask_path, cropped_mask)
            success_count += 1

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            error_count += 1

    print(f"\nProcessing complete.")
    print(f"Successfully processed: {success_count}")
    print(f"Skipped (no matching mask): {skipped_count}")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='批量ROI提取工具')
    parser.add_argument('--mode', type=str, choices=['images', 'pairs'], default='images',
                        help='处理模式: images (仅处理图像) 或 pairs (处理图像-mask对)')
    parser.add_argument('--checkpoint', type=str,
                        default=r'E:\DSA\Research\test\ThyroidROI\outputs\best_dice_model.pth',
                        help='模型权重路径')

    # For images mode(单一图片集模式)
    parser.add_argument('--input_dir', type=str,
                        default=r'E:\DSA\Research\Thyroid\Code\MetricsPileline\[353]ReportData',
                        help='输入图像目录')
    parser.add_argument('--output_dir', type=str,
                        default=r'E:\DSA\Research\Thyroid\Code\MetricsPileline\[353]ReportData_ROI',
                        help='输出图像目录')

    # For pairs mode(图片-mask集模式)
    parser.add_argument('--images_dir', type=str,
                        default=r'E:\DSA\Research\test\ThyroidROI\PKTN\images',
                        help='输入图像目录 ')
    parser.add_argument('--masks_dir', type=str,
                        default=r'E:\DSA\Research\test\ThyroidROI\PKTN\masks',
                        help='输入mask目录 ')
    parser.add_argument('--output_images_dir', type=str,
                        default=r'E:\DSA\Research\test\ThyroidROI\PKTN_ROI\images',
                        help='输出图像目录 ')
    parser.add_argument('--output_masks_dir', type=str,
                        default=r'E:\DSA\Research\test\ThyroidROI\PKTN_ROI\masks',
                        help='输出mask目录 ')

    args = parser.parse_args()

    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if args.mode == 'images':
        # Single images processing mode
        print("=" * 60)
        print("模式: 单独处理图像")
        print("=" * 60)

        if not os.path.exists(args.input_dir):
            print(f"Error: Input directory not found: {args.input_dir}")
            sys.exit(1)

        batch_process(args.input_dir, args.output_dir, args.checkpoint)

    elif args.mode == 'pairs':
        # Image-mask pairs processing mode
        print("=" * 60)
        print("模式: 处理图像-mask对")
        print("=" * 60)

        if not os.path.exists(args.images_dir):
            print(f"Error: Images directory not found: {args.images_dir}")
            sys.exit(1)

        if not os.path.exists(args.masks_dir):
            print(f"Error: Masks directory not found: {args.masks_dir}")
            sys.exit(1)

        batch_process_with_masks(
            args.images_dir,
            args.masks_dir,
            args.output_images_dir,
            args.output_masks_dir,
            args.checkpoint
        )

    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
