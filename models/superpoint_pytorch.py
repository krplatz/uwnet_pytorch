# superpoint_pytorch.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace

class SuperPointDense(nn.Module):
    """
    PyTorch implementation of SuperPoint that matches the original architecture
    from Magic Leap paper. Modified to return dense maps instead of sparse points.
    
    Returns:
      'dense_score' -> (N,1,H_feat,W_feat)
      'dense_desc'  -> (N,256,H_feat,W_feat)
    exactly matching a 1/8 resolution if your CNN uses stride=8.
    """
    
    def __init__(self):
        super(SuperPointDense, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        
        # Shared Encoder (exactly matching original SuperPointNet)
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)
        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)
        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)
        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)
        
        # Detector Head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)
        
        # Descriptor Head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        Forward pass that returns dense feature maps.
        Input:
          x: Image tensor shaped N x 1 x H x W or N x 3 x H x W (will be converted to grayscale)
        Output:
          Dictionary with 'dense_score' and 'dense_desc'
        """
        # Convert to grayscale if needed
        if x.shape[1] == 3:
            # Convert RGB -> grayscale
            scale = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            x = (x * scale).sum(1, keepdim=True)
            
        # Shared Encoder - exact same architecture as original SuperPoint
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        
        # Detector Head
        cPa = self.relu(self.convPa(x))
        semi = self.convPb(cPa)  # (N,65,H/8,W/8)
        
        # Convert semi to score map for distillation
        prob = F.softmax(semi, dim=1)  # (N,65,H/8,W/8)
        # Skip the last background channel
        prob_nobg = prob[:, :-1]  # (N,64,H/8,W/8)
        # Average across subpixel channels to get a dense score map
        score_map = prob_nobg.mean(dim=1, keepdim=True)  # (N,1,H/8,W/8)
        
        # Descriptor Head
        cDa = self.relu(self.convDa(x))
        desc = self.convDb(cDa)  # (N,256,H/8,W/8)
        desc = F.normalize(desc, p=2, dim=1)  # L2 normalize descriptors
        
        return {
            "dense_score": score_map,  # shape (N,1,H/8,W/8)
            "dense_desc": desc,        # shape (N,256,H/8,W/8)
        }


# Keep the VGGBlock for other parts of your code that might use it
class VGGBlock(nn.Sequential):
    def __init__(self, c_in, c_out, kernel_size, relu=True):
        padding = (kernel_size - 1)//2
        conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding)
        bn   = nn.BatchNorm2d(c_out, eps=0.001)
        activation = nn.ReLU(inplace=True) if relu else nn.Identity()
        super().__init__(conv, activation, bn)
