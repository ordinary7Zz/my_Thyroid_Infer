import os
import sys
import shutil
import cv2
import numpy as np
from tqdm import tqdm
import torch

# Add ThyroidROI to path to import ROIExtractor
# sys.path.append(r'E:\DSA\Research\Thyroid\Code\ThyroidROI')
from roi_extractor import ROIExtractor

def batch_process(input_root, output_root, checkpoint_path):
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

if __name__ == "__main__":
    # Define paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    #input_dir = os.path.join(current_dir, "SelectedReport")
    #output_dir = os.path.join(current_dir, "SelectedReport_ROI")
    input_dir = r"/mnt/wangbd8/workspace/DataSets/ThyroidAgent/PKTN/masks"
    output_dir = r"/mnt/wangbd8/workspace/DataSets/ThyroidAgent/PKTN_processed/masks"
    checkpoint = r"/mnt/wangbd8/workspace/ThyroidAgent/ThyroidROI/outputs/best_dice_model.pth"
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
        
    batch_process(input_dir, output_dir, checkpoint)
