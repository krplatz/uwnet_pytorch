import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class PairedDataset(Dataset):
    """
    Loads pairs of RGB air images (from SUNRGBD) and their corresponding 
    pseudo-underwater versions to ensure proper correspondence during training.
    """
    def __init__(self, sun_root_dir, pseudo_dir, transform=None, max_samples=None):
        self.sun_root_dir = sun_root_dir
        self.pseudo_dir = pseudo_dir
        self.transform = transform
        self.samples = []
        
        # Assume the extra folder is called "NYUdata"
        nyu_data_dir = os.path.join(sun_root_dir, "NYUdata")
        if not os.path.isdir(nyu_data_dir):
            raise RuntimeError(f"Expected folder 'NYUdata' in {sun_root_dir}")
        
        # Get list of all pseudo-underwater images
        pseudo_files = sorted([f for f in os.listdir(pseudo_dir) 
                              if f.lower().endswith((".png", ".jpg"))])
        pseudo_basenames = [os.path.splitext(f)[0] for f in pseudo_files]
        
        # Count of skipped images (no matching pseudo)
        skipped_count = 0
        
        # Walk through each subject folder inside NYUdata (e.g., NYU0001, NYU0002, ...)
        for subject in sorted(os.listdir(nyu_data_dir)):
            subject_path = os.path.join(nyu_data_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            
            # Extract the base name (e.g., "NYU0001") to match with pseudo files
            subject_basename = subject
            
            # Skip if we don't have a corresponding pseudo-underwater image
            if subject_basename not in pseudo_basenames:
                skipped_count += 1
                continue  # Removed the warning message
            
            # Get the paths to air and depth images
            image_dir = os.path.join(subject_path, "image")
            depth_dir = os.path.join(subject_path, "depth_bfx")
            if not os.path.isdir(image_dir) or not os.path.isdir(depth_dir):
                continue
            
            # Get the corresponding air image
            image_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.lower().endswith((".png", ".jpg"))])
            if not image_files:
                continue
            
            # Use the first image in the directory (should only be one)
            air_img_path = os.path.join(image_dir, image_files[0])
            
            # Find the corresponding pseudo-underwater image
            pseudo_img_filename = pseudo_basenames.index(subject_basename)
            pseudo_img_path = os.path.join(pseudo_dir, pseudo_files[pseudo_img_filename])
            
            self.samples.append((air_img_path, pseudo_img_path))
        
        print(f"Found {len(self.samples)} paired air-pseudo images (skipped {skipped_count} without matches)")
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        air_img_path, pseudo_img_path = self.samples[idx]
        air_img = cv2.imread(air_img_path, cv2.IMREAD_COLOR)
        pseudo_img = cv2.imread(pseudo_img_path, cv2.IMREAD_COLOR)
        
        if air_img is None or pseudo_img is None:
            raise RuntimeError(f"Failed to load images: {air_img_path}, {pseudo_img_path}")
        
        if self.transform:
            air_img = self.transform(air_img)
            pseudo_img = self.transform(pseudo_img)
            
        return {'air_img': air_img, 'pseudo_img': pseudo_img}

def get_paired_dataloader(sun_root_dir, pseudo_dir, batch_size=4, shuffle=True, max_samples=None, num_workers=2):
    transform_resize = T.Compose([
        T.ToPILImage(),
        T.Resize((480, 640)),  # pick your multiple-of-8 shape
        T.ToTensor()
    ])

    dataset = PairedDataset(sun_root_dir, pseudo_dir, transform=transform_resize, max_samples=max_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)