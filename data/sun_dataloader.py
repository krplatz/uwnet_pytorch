import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class SUNDatasetNested(Dataset):
    """
    Loads RGB (air) and depth images from a nested SUNRGBD-like directory structure.
    
    Expected structure:
    
    SUNRGBD/
      NYUdata/
        NYU0001/
          image/
            NYU0001.jpg
            ...
          depth/
            NYU0001.png
            ...
        NYU0002/
          image/
            ...
          depth/
            ...
    """
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        # Assume the extra folder is called "NYUdata"
        nyu_data_dir = os.path.join(root_dir, "NYUdata")
        if not os.path.isdir(nyu_data_dir):
            raise RuntimeError(f"Expected folder 'NYUdata' in {root_dir}")
        
        # Walk through each subject folder inside NYUdata (e.g., NYU0001, NYU0002, ...)
        for subject in sorted(os.listdir(nyu_data_dir)):
            subject_path = os.path.join(nyu_data_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            
            image_dir = os.path.join(subject_path, "image")
            depth_dir = os.path.join(subject_path, "depth_bfx")
            if not os.path.isdir(image_dir) or not os.path.isdir(depth_dir):
                continue
            
            image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg"))])
            depth_files = sorted([f for f in os.listdir(depth_dir) if f.lower().endswith((".png", ".jpg"))])
            
            # Match files based on base name (e.g., "NYU0001")
            for img_file in image_files:
                base_name = os.path.splitext(img_file)[0]
                # Try .png first; if not, try .jpg for depth file
                possible_depth_file = base_name + ".png"
                if possible_depth_file not in depth_files:
                    possible_depth_file = base_name + ".jpg"
                if possible_depth_file in depth_files:
                    img_path = os.path.join(image_dir, img_file)
                    depth_path = os.path.join(depth_dir, possible_depth_file)
                    self.samples.append((img_path, depth_path))
        
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        air_img_path, depth_img_path = self.samples[idx]
        rgb_img = cv2.imread(air_img_path, cv2.IMREAD_COLOR)
        depth_img = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)
        if rgb_img is None or depth_img is None:
            raise RuntimeError(f"Failed to load images: {air_img_path}, {depth_img_path}")
        # Convert depth image to meters (assuming stored in millimeters)
        depth_img = depth_img.astype(np.float32) / 1000.0
        if self.transform:
            rgb_img = self.transform(rgb_img)
        return {'air_img': rgb_img, 'depth_map': depth_img}

def get_sun_dataloader(root_dir, batch_size=4, shuffle=True, max_samples=None):
    transform_resize = T.Compose([
        T.ToPILImage(),
        T.Resize((480, 640)),  # pick your multiple-of-8 shape
        T.ToTensor()
    ])

    dataset = SUNDatasetNested(root_dir, transform=transform_resize, max_samples=max_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
