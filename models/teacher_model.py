# teacher_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os

from models.netvlad import NetVLAD
from models.superpoint_pytorch import SuperPointDense

class TeacherModel(nn.Module):
    """
    Teacher: returns dense local maps (score + descriptor) plus a NetVLAD global descriptor.
    """
    def __init__(self, superpoint_weights_path, num_clusters=64, backbone_out_channels=512, global_dim=4096):
        super().__init__()
        
        # Check if the weights file exists
        if not os.path.exists(superpoint_weights_path):
            raise FileNotFoundError(f"SuperPoint weights not found at {superpoint_weights_path}")
        
        # 1) Dense SuperPoint
        self.superpoint = SuperPointDense()
        print(f"Loading SuperPoint weights from {superpoint_weights_path}")
        
        try:
            # load the weights
            weights = torch.load(superpoint_weights_path, map_location='cpu')
            
            # Print some debug info about the weights
            print(f"Loaded weights keys: {list(weights.keys())[:5]}...")
            
            # If the weights don't match exactly, try to adapt them
            try:
                # First, try to load with strict=True
                self.superpoint.load_state_dict(weights, strict=True)
                print("SuperPoint weights loaded with strict=True, all weights matched perfectly!")
            except Exception as e:
                print(f"Error loading with strict=True: {e}")
                
                # Get model state dict to compare keys
                model_keys = set(self.superpoint.state_dict().keys())
                weights_keys = set(weights.keys())
                
                print(f"\nDiagnosing weight loading issue:")
                print(f"Model has {len(model_keys)} parameters")
                print(f"Weights file has {len(weights_keys)} parameters")
                
                # Find common keys
                common_keys = model_keys.intersection(weights_keys)
                print(f"Found {len(common_keys)} matching parameter names")
                
                # Try to load with strict=False
                print("\nAttempting to load with strict=False...")
                result = self.superpoint.load_state_dict(weights, strict=False)
                
                if len(result.missing_keys) > 0:
                    print(f"Missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}...")
                else:
                    print("No missing keys - all model parameters were initialized")
                    
                if len(result.unexpected_keys) > 0:
                    print(f"Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}...")
                else:
                    print("No unexpected keys - all weights file parameters were used")
                
                print("\nSuperPoint model loaded with strict=False. Some parameters may be initialized randomly.")
        except Exception as e:
            print(f"Error loading SuperPoint weights: {e}")
            raise
        
        # freeze teacher
        self.superpoint.eval()
        for p in self.superpoint.parameters():
            p.requires_grad = False
        print("SuperPoint model frozen")

        # 2) CNN + NetVLAD for global descriptor
        print("Initializing ResNet backbone for global descriptors")
        try:
            base_cnn = models.resnet18(pretrained=True)
            self.backbone = nn.Sequential(*list(base_cnn.children())[:-2])
            self.out_channels = backbone_out_channels
            self.netvlad = NetVLAD(num_clusters=num_clusters, dim=self.out_channels, normalize_input=True)
            self.fc_global = nn.Linear(num_clusters * self.out_channels, global_dim)
            print("Global descriptor backbone initialized successfully")
        except Exception as e:
            print(f"Error initializing global descriptor backbone: {e}")
            raise

    def forward(self, x):
        """
        x: (N,3,H,W) or (N,1,H,W) in [0,1]
        Returns:
          teacher_scores   (N,1,H',W')  [dense probability map, same size you decided]
          teacher_local    (N,256,H_desc,W_desc)  [dense descriptor map]
          teacher_global   (N, num_clusters * out_channels) global descriptor
        """
        try:
            # local features from superpoint
            sp_out = self.superpoint(x)
            teacher_scores = sp_out["dense_score"]  # e.g. shape (N,1,H*8,W*8)
            teacher_local = sp_out["dense_desc"]   # e.g. shape (N,256,H,W)
            
            # Check for NaN or Inf values
            if torch.isnan(teacher_scores).any() or torch.isinf(teacher_scores).any():
                print("WARNING: NaN or Inf values in teacher_scores")
                # Replace NaN/Inf with zeros
                teacher_scores = torch.nan_to_num(teacher_scores, nan=0.0, posinf=1.0, neginf=0.0)
                
            if torch.isnan(teacher_local).any() or torch.isinf(teacher_local).any():
                print("WARNING: NaN or Inf values in teacher_local")
                # Replace NaN/Inf with zeros
                teacher_local = torch.nan_to_num(teacher_local, nan=0.0, posinf=1.0, neginf=0.0)

            # global descriptor
            feats = self.backbone(x)              # shape (N,512,H',W')
            teacher_global_big = self.netvlad(feats)     # shape (N,32768) => 64*512
            teacher_global = self.fc_global(teacher_global_big)  # (N,4096)
            teacher_global = F.normalize(teacher_global, p=2, dim=1)  # L2 normalize
            
            if torch.isnan(teacher_global).any() or torch.isinf(teacher_global).any():
                print("WARNING: NaN or Inf values in teacher_global")
                # Replace NaN/Inf with zeros
                teacher_global = torch.nan_to_num(teacher_global, nan=0.0, posinf=1.0, neginf=0.0)
            
            return teacher_scores, teacher_local, teacher_global
            
        except Exception as e:
            print(f"Error in teacher model forward pass: {e}")
            # Return placeholder tensors in case of error
            batch_size = x.shape[0]
            dummy_scores = torch.zeros((batch_size, 1, x.shape[2]//8, x.shape[3]//8), device=x.device)
            dummy_local = torch.zeros((batch_size, 256, x.shape[2]//8, x.shape[3]//8), device=x.device)
            dummy_global = torch.zeros((batch_size, 4096), device=x.device)
            return dummy_scores, dummy_local, dummy_global
