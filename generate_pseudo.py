import os
import cv2
import numpy as np
from utils.pseudo_underwater import pseudo_underwater_generator
from data.sun_dataloader import SUNDatasetNested

# Update this path to your SUNRGBD folder (which now contains a "NYUdata" folder)
sun_root = r"../SUNRGBD"

# Output directory for pseudo-underwater images
output_dir = "pseudo_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create the dataset using the nested structure
dataset = SUNDatasetNested(root_dir=sun_root, max_samples=1000)  # adjust max_samples as needed

print(f"Found {len(dataset)} samples.")

# Loop through each sample and generate pseudo-underwater images
for idx in range(len(dataset)):
    sample = dataset[idx]
    air_img = sample['air_img']       # Original air image (H, W, 3)
    depth_map = sample['depth_map']     # Depth map (H, W) in meters

    pseudo_img = pseudo_underwater_generator(
        air_img,
        depth_map,
        randomize=True
    )

    # Save the pseudo image using the base filename of the air image
    base_filename = os.path.splitext(os.path.basename(dataset.samples[idx][0]))[0]
    filename = base_filename + ".png"
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, pseudo_img)
    print(f"Saved {save_path}")

print("Pseudo-underwater image generation complete.")
