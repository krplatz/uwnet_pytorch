import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

class PseudoImageDataset(Dataset):
    """
    Loads pre-generated pseudo-underwater images from a flat folder.
    """
    def __init__(self, pseudo_dir, transform=None, max_samples=None):
        self.pseudo_dir = pseudo_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(pseudo_dir) if f.endswith('.png') or f.endswith('.jpg')])
        if max_samples is not None:
            self.image_files = self.image_files[:max_samples]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.pseudo_dir, img_filename)
        pseudo_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if pseudo_img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        if self.transform:
            pseudo_img = self.transform(pseudo_img)
        return {'pseudo_img': pseudo_img}

def get_pseudo_dataloader(pseudo_dir, batch_size=4, shuffle=True, max_samples=None):
    transform_resize = T.Compose([
        T.ToPILImage(),
        T.Resize((480, 640)),
        T.ToTensor()
    ])

    dataset = PseudoImageDataset(pseudo_dir, transform=transform_resize, max_samples=max_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
