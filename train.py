# train.py
import torch
import torch.optim
import torch.nn as nn
from models.teacher_model import TeacherModel
from models.uwnet import UWNet
from data.sun_dataloader import get_sun_dataloader
from data.pseudo_dataloader import get_pseudo_dataloader
from data.paired_dataloader import get_paired_dataloader
from utils.losses import distillation_loss
import os
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train UWNet model with distillation.")
    parser.add_argument('--sun_root', type=str, default="../SUNRGBD", help="Path to the SUNRGBD dataset root directory")
    parser.add_argument('--pseudo_dir', type=str, default="pseudo_images", help="Path to the directory containing pseudo-underwater images")
    parser.add_argument('--superpoint_path', type=str, default="SuperPointPretrainedNetwork/superpoint_v1.pth", help="Path to SuperPoint pretrained weights .pth file")
    parser.add_argument('--epochs', type=int, default=150, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size.")
    parser.add_argument('--lr', type=float, default=0.003, help="Learning rate for RMSprop optimizer.")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("WARNING: No GPU found. Running on CPU will be significantly slower!")
        user_input = input("Continue with CPU? (y/n): ")
        if user_input.lower() != 'y':
            print("Training aborted. Please run on a system with GPU support.")
            return
    
    # 1) TEACHER with SuperPoint
    superpoint_path = args.superpoint_path
    if not os.path.exists(superpoint_path):
        print(f"Warning: SuperPoint weights not found at {superpoint_path}")
        print("Searching for weights file...")
        # Try to find it in the current directory or subdirectories as a fallback
        found_superpoint = False
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".pth") and "superpoint" in file.lower():
                    superpoint_path = os.path.join(root, file)
                    print(f"Found potential weights file: {superpoint_path}")
                    found_superpoint = True
                    break
            if found_superpoint:
                break
        if not found_superpoint:
            print(f"Could not find SuperPoint weights. Please check the path: {args.superpoint_path} or place them in the working directory.")
            return

    print(f"Loading teacher model with weights from: {superpoint_path}")
    try:
        teacher = TeacherModel(
            superpoint_weights_path=superpoint_path,
            num_clusters=64,
            backbone_out_channels=512
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print("Teacher model loaded successfully")
    except Exception as e:
        print(f"Error loading teacher model: {e}")
        return
    
    # 2) STUDENT = UWNet
    print("Initializing student model (UWNet)")
    student = UWNet().to(device)
    
    # 3) Data
    print("Loading datasets")
    sun_root = args.sun_root
    if not os.path.exists(sun_root):
        sun_root = input(f"SUNRGBD dataset not found at {sun_root} (from --sun_root argument). Please enter the correct path: ")
    
    pseudo_dir = args.pseudo_dir
    if not os.path.exists(pseudo_dir):
        pseudo_dir = input(f"Pseudo-underwater images not found at {pseudo_dir} (from --pseudo_dir argument). Please enter the correct path: ")
    
    try:
        # Use the paired dataloader to ensure correspondence
        paired_loader = get_paired_dataloader(sun_root, pseudo_dir, batch_size=args.batch_size, shuffle=True, max_samples=500)
        print(f"Loaded {len(paired_loader)} batches of paired air and pseudo images")
    except Exception as e:
        print(f"Error loading paired dataset: {e}")
        return
    
    # 4) Distillation
     # Initialize learnable log-variances (alpha, beta, gamma in the paper's loss eq)
    # Starting at 0.0 means initial weightings exp(0)=1 for L1, L2, 2*exp(0)=2 for L3 before regularizer
    log_alpha = nn.Parameter(torch.tensor(0.0, device=device))
    log_beta = nn.Parameter(torch.tensor(0.0, device=device))
    log_gamma = nn.Parameter(torch.tensor(0.0, device=device))
    print("Starting distillation training")
    optimizer = torch.optim.RMSprop(list(student.parameters()) + [log_alpha, log_beta, log_gamma], lr=args.lr)
    epochs = args.epochs
    
    # Training history for monitoring
    loss_history = []
    
    # Track the individual loss components
    global_loss_history = []
    local_loss_history = []
    score_loss_history = []
    
    # Create plot directory
    os.makedirs("training_plots", exist_ok=True)
    
    for epoch in range(epochs):
        student.train()
        ep_loss = 0.0
        ep_global_loss = 0.0
        ep_local_loss = 0.0
        ep_score_loss = 0.0
        batch_count = 0
        
        for batch in paired_loader:
            try:
                # Get the paired air and pseudo images from the same batch
                air_imgs = batch['air_img'].to(device)
                pseudo_imgs = batch['pseudo_img'].to(device)
                
                # Forward teacher (no grad)
                with torch.no_grad():
                    teacher_scores, teacher_local, teacher_global = teacher(air_imgs)
                
                # Forward student
                student_scores, student_local, student_global = student(pseudo_imgs)
                
                # Print tensor shapes for debugging
                if batch_count == 0 and epoch == 0:
                    print(f"Teacher scores shape: {teacher_scores.shape}")
                    print(f"Teacher local shape: {teacher_local.shape}")
                    print(f"Teacher global shape: {teacher_global.shape}")
                    print(f"Student scores shape: {student_scores.shape}")
                    print(f"Student local shape: {student_local.shape}")
                    print(f"Student global shape: {student_global.shape}")
                
                # Calculate individual loss components for monitoring
                global_loss = ((student_global - teacher_global)**2).mean()
                local_loss = ((student_local - teacher_local)**2).mean()
                score_loss = ((student_scores - teacher_scores)**2).mean()
                
                # Adaptive distillation loss using the learnable parameters
                loss = distillation_loss(
                            student_global, teacher_global,
                            student_local, teacher_local,
                            student_scores, teacher_scores,
                            log_alpha, log_beta, log_gamma # Pass the learnable parameters
                        )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track losses
                ep_loss += loss.item()
                ep_global_loss += global_loss.item()
                ep_local_loss += local_loss.item()
                ep_score_loss += score_loss.item()
                
                batch_count += 1
                if batch_count % 5 == 0:
                    print(f"  Batch {batch_count}/{len(paired_loader)}, Loss: {loss.item():.4f}")
                    print(f"    - Global loss: {global_loss.item():.4f}")
                    print(f"    - Local loss: {local_loss.item():.4f}")
                    print(f"    - Score loss: {score_loss.item():.4f}")
            
            except Exception as e:
                print(f"Error in batch processing: {e}")
                continue
        
        # Record history
        avg_loss = ep_loss/batch_count
        avg_global_loss = ep_global_loss/batch_count
        avg_local_loss = ep_local_loss/batch_count
        avg_score_loss = ep_score_loss/batch_count
        
        loss_history.append(avg_loss)
        global_loss_history.append(avg_global_loss)
        local_loss_history.append(avg_local_loss)
        score_loss_history.append(avg_score_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss={avg_loss:.4f}")
        print(f"  - Global loss: {avg_global_loss:.4f}")
        print(f"  - Local loss: {avg_local_loss:.4f}")
        print(f"  - Score loss: {avg_score_loss:.4f}")
        
        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            checkpoint_path = f"UWNet_v9_epoch{epoch+1}.pth"
            torch.save(student.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Plot and save training curves
            plt.figure(figsize=(12, 8))
            
            # Plot overall loss
            plt.subplot(2, 2, 1)
            plt.plot(loss_history, 'b-', label='Total Loss')
            plt.title('Overall Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Plot component losses
            plt.subplot(2, 2, 2)
            plt.plot(global_loss_history, 'r-', label='Global Loss')
            plt.plot(local_loss_history, 'g-', label='Local Loss')
            plt.plot(score_loss_history, 'y-', label='Score Loss')
            plt.title('Component Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(f"training_plots/training_curves_epoch{epoch+1}.png")
            plt.close()

    # Save final student
    save_path = "UWNet_v9.pth"
    print(f"Saving trained model to {save_path}")
    torch.save(student.state_dict(), save_path)
    
    # Plot final training curves
    plt.figure(figsize=(12, 8))
    
    # Plot overall loss
    plt.subplot(2, 2, 1)
    plt.plot(loss_history, 'b-', label='Total Loss')
    plt.title('Overall Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot component losses
    plt.subplot(2, 2, 2)
    plt.plot(global_loss_history, 'r-', label='Global Loss')
    plt.plot(local_loss_history, 'g-', label='Local Loss')
    plt.plot(score_loss_history, 'y-', label='Score Loss')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig("training_plots/final_training_curves.png")
    
    # Print training summary
    print("\nTraining Summary:")
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Final component losses:")
    print(f"  - Global: {global_loss_history[-1]:.4f}")
    print(f"  - Local: {local_loss_history[-1]:.4f}")
    print(f"  - Score: {score_loss_history[-1]:.4f}")
    print("Training complete!")

if __name__ == "__main__":
    main()
