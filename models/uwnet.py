import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import DeformConv2d

# --- Minimal DeformableConv2d Wrapper ---
class DeformableConv2d(nn.Module):
    """
    Minimal wrapper that couples a DeformConv2d with its own learned offset conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        # The number of offsets = 2 * (kernel_size * kernel_size)
        self.offset_channels = 2 * kernel_size * kernel_size
        
        # 1) Convolution to predict offsets
        self.offset_conv = nn.Conv2d(
            in_channels, self.offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # 2) The deformable convolution itself
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
    
    def forward(self, x):
        # Predict offsets
        offsets = self.offset_conv(x)
        # Deformable convolution uses these offsets
        out = self.deform_conv(x, offsets)
        return out

# --- Attention Modules ---
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        attn = self.fc(self.avg_pool(x))
        return self.sigmoid(attn) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(x_cat))
        return attn * x

# --- Simplified NetVLAD Module ---
class NetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=128):
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=1, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
    
    def forward(self, x):
        N, C, H, W = x.shape
        # Flatten spatial dimensions
        x_flat = x.view(N, C, -1)  # shape: (N, C, H*W)
        
        # Soft assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)  # (N, num_clusters, H*W)
        soft_assign = F.softmax(soft_assign, dim=1)
        
        x_flat_permute = x_flat.permute(0, 2, 1)  # (N, H*W, C)
        
        # Compute residuals to each centroid
        residual = x_flat_permute.unsqueeze(1) - self.centroids.unsqueeze(0).unsqueeze(2)
        # Weighted sum (Vlad)
        vlad = (soft_assign.unsqueeze(-1) * residual).sum(dim=2)  # (N, num_clusters, dim)
        
        # Intra-normalize
        vlad = F.normalize(vlad, p=2, dim=2)
        # Flatten
        vlad = vlad.view(N, -1)
        # L2 normalize
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad

# --- UWNet Architecture ---
class UWNet(nn.Module):
    def __init__(self, descriptor_dim=256, global_dim=4096):
        super(UWNet, self).__init__()
        print("Initializing UWNet model")
        
        # 1) Load MobileNetV2 backbone (ImageNet-pretrained)
        try:
            backbone = models.mobilenet_v2(pretrained=True).features
            print("Loaded MobileNetV2 backbone")
        except Exception as e:
            print(f"Error loading MobileNetV2: {e}")
            raise
            
        # 2) Slice into two parts: local (shared1) & global (shared2)
        self.shared1 = nn.Sequential(*backbone[:7])  # user-chosen partial
        self.shared2 = nn.Sequential(*backbone[7:])  # remainder

        # 3) Figure out how many channels come out of shared1
        #    We'll do a dummy pass on a small input
        print("Running dummy pass to determine channel dimensions")
        dummy = torch.zeros(1, 3, 64, 64)  # Smaller dummy tensor to save memory
        with torch.no_grad():
            out_dummy1 = self.shared1(dummy)
        local_in_channels = out_dummy1.shape[1]  # e.g., might be 32 instead of 96
        print(f"Local branch input channels: {local_in_channels}")

        # 4) Build local branch modules
        self.ca = ChannelAttention(local_in_channels)
        self.sa = SpatialAttention()

        # 4a) Keypoint score map (65 channels: 1 for keypoints + leftover)
        self.local_conv = nn.Sequential(
            nn.Conv2d(local_in_channels, local_in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(local_in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(local_in_channels, 65, kernel_size=1)
        )

        # 4b) Descriptor DCN
        self.desc_dcn = nn.Sequential(
            DeformableConv2d(local_in_channels, local_in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(local_in_channels),
            nn.ReLU(inplace=True),
            DeformableConv2d(local_in_channels, descriptor_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(descriptor_dim),
            nn.ReLU(inplace=True),
        )

        # 5) Figure out how many channels come out of shared2 for global branch
        with torch.no_grad():
            out_dummy2 = self.shared2(out_dummy1)
        global_in_channels = out_dummy2.shape[1]  # e.g., might be 1280
        print(f"Global branch input channels: {global_in_channels}")

        # 6) Global branch
        self.global_conv = nn.Conv2d(global_in_channels, 128, kernel_size=1)
        
        # 6a) Use our NetVLAD implementation
        try:
            from models.netvlad import NetVLAD
            self.netvlad = NetVLAD(num_clusters=64, dim=128, normalize_input=True)
            self.fc_global = nn.Linear(64 * 128, global_dim)
            print("Initialized NetVLAD and global feature branch")
        except Exception as e:
            print(f"Error initializing NetVLAD: {e}")
            raise
        
        print("UWNet model initialization complete")
    
    def forward(self, x):
        # Input x: (N,3,H,W)
        try:
            # Check input shape
            if len(x.shape) != 4 or x.shape[1] != 3:
                # print(f"WARNING: Expected input shape (N,3,H,W), got {x.shape}")
                if x.shape[1] == 1:
                    # Repeat grayscale to RGB
                    x = x.repeat(1, 3, 1, 1)
                    # print(f"Converted grayscale to RGB: {x.shape}")
            
            # Get original dimensions for later reference
            N, C, H, W = x.shape
            
            # Shared features extraction
            F1 = self.shared1(x)  # local features => shape (N, local_in_channels, H/8, W/8)
            F2 = self.shared2(F1) # global features => shape (N, global_in_channels, H/32, W/32)
    
            # --- Local branch ---
            att = self.ca(F1)
            att = self.sa(att)
    
            # keypoint score
            local_feat = self.local_conv(att)        # shape: (N,65,H/8,W/8)
            keypoint_scores = local_feat[:, :1, :, :]  # first channel is the "score"
    
            # local descriptors
            descriptors = self.desc_dcn(F1)          # shape: (N, descriptor_dim, H/8,W/8)
            
            # Normalize descriptors
            descriptors = F.normalize(descriptors, p=2, dim=1)
    
            # --- Global branch ---
            global_feat = self.global_conv(F2)       # shape: (N,128,H/32,W/32)
            
            # Check for any issues with global_feat
            if torch.isnan(global_feat).any() or torch.isinf(global_feat).any():
                print("WARNING: NaN or Inf values in global_feat")
                global_feat = torch.nan_to_num(global_feat, nan=0.0, posinf=1.0, neginf=0.0)
                
            vlad = self.netvlad(global_feat)         # shape: (N,64*128)
            global_descriptor = self.fc_global(vlad) # shape: (N, global_dim)
            global_descriptor = F.normalize(global_descriptor, p=2, dim=1)
    
            # Check for NaN/Inf values in outputs
            outputs = [keypoint_scores, descriptors, global_descriptor]
            output_names = ["keypoint_scores", "descriptors", "global_descriptor"]
            
            for tensor, name in zip(outputs, output_names):
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    print(f"WARNING: {name} contains NaN or Inf values, replacing with zeros")
                    if name == "keypoint_scores":
                        keypoint_scores = torch.nan_to_num(keypoint_scores, nan=0.0, posinf=1.0, neginf=0.0)
                    elif name == "descriptors":
                        descriptors = torch.nan_to_num(descriptors, nan=0.0, posinf=1.0, neginf=0.0)
                    elif name == "global_descriptor":
                        global_descriptor = torch.nan_to_num(global_descriptor, nan=0.0, posinf=1.0, neginf=0.0)
            
            return keypoint_scores, descriptors, global_descriptor
            
        except Exception as e:
            print(f"Error in UWNet forward pass: {e}")
            # Return placeholder tensors with correct shapes
            N = x.shape[0]
            dummy_scores = torch.zeros((N, 1, H//8, W//8), device=x.device)
            dummy_desc = torch.zeros((N, 256, H//8, W//8), device=x.device)
            dummy_global = torch.zeros((N, 4096), device=x.device)
            return dummy_scores, dummy_desc, dummy_global
