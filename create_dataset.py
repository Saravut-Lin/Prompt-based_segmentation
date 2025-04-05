#!/usr/bin/env python3
import os
import glob
import shutil
import random
from PIL import Image
import numpy as np

# Create the structure that SAM2 expects when single_object_mode=false
# For Oxford Pets, we need a structure where:
# - Each image name is treated as a "video"
# - Each image/mask goes in its own subfolder

# Paths
OXFORD_PATH = "/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/dataset/oxford-iiit-pet"
IMAGES_PATH = os.path.join(OXFORD_PATH, "images")
TRIMAP_PATH = os.path.join(OXFORD_PATH, "annotations", "trimaps")

# Output directories
OUT_ROOT = "/Users/saravut_lin/EDINBURGH/Semester_2/ComV/Mini-Project/sam2/data/oxford_pets"
OUT_IMAGES = os.path.join(OUT_ROOT, "JPEGImages")
OUT_MASKS = os.path.join(OUT_ROOT, "Annotations")

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_MASKS, exist_ok=True)

# Process all images and masks
all_files = []

# Get all image files
image_files = glob.glob(os.path.join(IMAGES_PATH, "*.jpg"))
print(f"Found {len(image_files)} images")

for img_path in image_files:
    # Get base filename
    basename = os.path.basename(img_path)
    filename = os.path.splitext(basename)[0]
    
    # Create directories for this "video"
    video_dir = os.path.join(OUT_IMAGES, filename)
    mask_dir = os.path.join(OUT_MASKS, filename)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Copy image to its video folder with frame ID 0
    dst_img = os.path.join(video_dir, "00000.jpg")
    shutil.copy(img_path, dst_img)
    
    # Convert and save mask
    trimap_path = os.path.join(TRIMAP_PATH, f"{filename}.png")
    if os.path.exists(trimap_path):
        # Convert trimap to binary mask
        trimap = np.array(Image.open(trimap_path))
        binary_mask = np.zeros_like(trimap, dtype=np.uint8)
        binary_mask[trimap == 1] = 1  # foreground
        binary_mask[trimap == 3] = 1  # boundary
        
        # Save binary mask with frame ID 0
        dst_mask = os.path.join(mask_dir, "00000.png")
        Image.fromarray(binary_mask * 255).save(dst_mask)
        
        all_files.append(filename)
    else:
        print(f"Warning: No mask found for {filename}")

# Create train/val splits
random.seed(42)
random.shuffle(all_files)
split_idx = int(len(all_files) * 0.8)
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

# Write train/val files
with open(os.path.join(OUT_ROOT, "train_list.txt"), "w") as f:
    f.write("\n".join(train_files))

with open(os.path.join(OUT_ROOT, "val_list.txt"), "w") as f:
    f.write("\n".join(val_files))

# Create tiny test subsets
with open(os.path.join(OUT_ROOT, "train_list_tiny.txt"), "w") as f:
    f.write("\n".join(train_files[:10]))

with open(os.path.join(OUT_ROOT, "val_list_tiny.txt"), "w") as f:
    f.write("\n".join(val_files[:5]))

print(f"Created train split with {len(train_files)} images")
print(f"Created validation split with {len(val_files)} images")
print(f"Created tiny subsets for testing")
print("Dataset preparation complete!")