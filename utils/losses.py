import torch
import torch.nn.functional as F

def distillation_loss(student_global, teacher_global, 
                        student_local, teacher_local, 
                        student_scores, teacher_scores,
                        alpha=0.0, beta=0.0, gamma=0.0):
    """
    Compute the knowledge distillation loss as described in Equation (8) of the paper:
    
      L = exp(-α) * L1 + exp(-β) * L2 + 2 * exp(-γ) * L3 + (α + β + γ)
      
    where:
      L1 = 0.5 * || student_global - teacher_global ||_2^2
      L2 = 0.5 * || student_local - teacher_local ||_2^2
      L3 = Mean Squared Error between student and teacher keypoint score maps
      
    Args:
        student_global (Tensor): Global descriptors from the student network.
        teacher_global (Tensor): Global descriptors from the teacher network.
        student_local (Tensor): Local descriptors from the student network.
        teacher_local (Tensor): Local descriptors from the teacher network.
        student_scores (Tensor): Keypoint score map from the student network.
        teacher_scores (Tensor): Keypoint score map from the teacher network.
        alpha (float): Weighting parameter for global loss term.
        beta (float): Weighting parameter for local loss term.
        gamma (float): Weighting parameter for score loss term.
        
    Returns:
        Tensor: The combined distillation loss.
    """
    
    # Handle potential shape mismatches by resizing tensors if needed
    if student_local.shape != teacher_local.shape:
        print(f"WARNING: Local descriptor shape mismatch: student={student_local.shape}, teacher={teacher_local.shape}")
        # Resize student to match teacher if possible
        if len(student_local.shape) == 4 and len(teacher_local.shape) == 4:
            student_local = F.interpolate(student_local, size=(teacher_local.shape[2], teacher_local.shape[3]), 
                                         mode='bilinear', align_corners=False)
    
    if student_scores.shape != teacher_scores.shape:
        print(f"WARNING: Score shape mismatch: student={student_scores.shape}, teacher={teacher_scores.shape}")
        # Resize student to match teacher if possible
        if len(student_scores.shape) == 4 and len(teacher_scores.shape) == 4:
            student_scores = F.interpolate(student_scores, size=(teacher_scores.shape[2], teacher_scores.shape[3]), 
                                          mode='bilinear', align_corners=False)
    
    # Loss 1: Global descriptor loss (L2 distance between global feature vectors)
    L1 = 0.5 * F.mse_loss(student_global, teacher_global)
    
    # Loss 2: Local descriptor loss (L2 distance between local feature maps)
    L2 = 0.5 * F.mse_loss(student_local, teacher_local)
    
    # Loss 3: Keypoint score loss (MSE between score maps)
    L3 = F.mse_loss(student_scores, teacher_scores)
    
    # Combined loss with automatic weighting (Equation 8)
    # L = e^(-α) * L1 + e^(-β) * L2 + 2 * e^(-γ) * L3 + (α + β + γ)
    loss = torch.exp(-alpha) * L1 + torch.exp(-beta) * L2 + 2 * torch.exp(-gamma) * L3 + (alpha + beta + gamma)
    
    # Print intermediate loss values for debugging
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"WARNING: Loss is {loss} - Components: L1={L1}, L2={L2}, L3={L3}")
        
        # Check for NaN values in inputs
        inputs = [student_global, teacher_global, student_local, teacher_local, student_scores, teacher_scores]
        input_names = ["student_global", "teacher_global", "student_local", "teacher_local", "student_scores", "teacher_scores"]
        
        for tensor, name in zip(inputs, input_names):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"WARNING: {name} contains NaN or Inf values")
        
        # Return a safe loss tensor on the same device as the inputs
        safe_device = student_global.device
        return torch.tensor(0.1, device=safe_device, requires_grad=True)
    
    return loss
