import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.uwnet import UWNet
from PIL import Image
import os
import argparse

cap = cv2.VideoCapture(cv2.CAP_DSHOW)


def load_image(image_path):
    """Load an image and convert to tensor"""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img) / 255.0  # Normalize to [0,1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return img, img_tensor

def get_webcam_frame():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = frame.convert('RGB')
    frame = np.array(frame) / 255.0
    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().unsqueeze(0)
    return frame, frame_tensor


def detect_keypoints(model, img_tensor, threshold=0.015):
    """Run inference and extract keypoints above threshold"""
    with torch.no_grad():
        scores, descriptors, _ = model(img_tensor)
    
    # Convert to numpy for processing
    scores_np = scores.squeeze().cpu().numpy()

    # -- NEW: Mask out the top region --
    TOP_MARGIN = 2
    # Make a boolean mask that's True everywhere except the top region
    mask = np.ones_like(scores_np, dtype=bool)
    mask[:TOP_MARGIN, :] = False  # zero out the top portion
    scores_np[~mask] = 0.0       # set those scores to zero
    
    # Find keypoints above threshold
    keypoints = []
    coords = np.where(scores_np > threshold)
    scores_filtered = scores_np[coords]
    
    # Sort by score for better visualization
    sort_idx = np.argsort(scores_filtered)[::-1]
    coords_y = coords[0][sort_idx]
    coords_x = coords[1][sort_idx]
    scores_sorted = scores_filtered[sort_idx]
    
    # Scale coordinates back to original image size
    h, w = img_tensor.shape[2:]
    img_h, img_w = img_tensor.shape[2:]
    scale_x = img_w / scores_np.shape[1]
    scale_y = img_h / scores_np.shape[0]
    
    for i in range(len(coords_y)):
        y = int(coords_y[i] * scale_y)
        x = int(coords_x[i] * scale_x)
        score = scores_sorted[i]
        keypoints.append((x, y, score))
    
    return keypoints, scores_np

def visualize_keypoints(img, keypoints, scores_np, max_points=500):
    """Visualize keypoints on the image"""
    # Create a heat map from scores
    heatmap = cv2.normalize(scores_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Create a blended image
    img_vis = (img * 255).astype(np.uint8)
    blended = cv2.addWeighted(img_vis, 0.7, heatmap, 0.3, 0)
    
    # Draw keypoints
    keypoints = keypoints[:min(len(keypoints), max_points)]
    for x, y, score in keypoints:
        color = (0, 255, 0)  # Green
        size = int(3 + 2 * score * 10)  # Size based on score
        cv2.circle(blended, (x, y), size, color, 1)
    
    return blended, heatmap

def main():
    parser = argparse.ArgumentParser(description='Test keypoint detection with UWNet')
    parser.add_argument('--model_path', type=str, default='UWNet_v7.pth', 
                        help='Path to trained UWNet model')
    parser.add_argument('--image_dir', type=str, default='pseudo_images', 
                        help='Directory containing images to test')
    parser.add_argument('--threshold', type=float, default=0.0015, 
                        help='Keypoint detection threshold')
    parser.add_argument('--num_images', type=int, default=3, 
                        help='Number of images to test')
    parser.add_argument('--output_dir', type=str, default='data_visualization/results', 
                        help='Directory to save results')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = UWNet().to(device)
    print(f"Loading model from {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Get list of images
    image_files = [f for f in os.listdir(args.image_dir) if f.endswith('.png')]
    image_files = image_files[:args.num_images]
    
    if not image_files:
        print(f"No images found in {args.image_dir}")
        return
    
    # Process each image
    fig, axes = plt.subplots(len(image_files), 3, figsize=(15, 5 * len(image_files)))
    if len(image_files) == 1:
        axes = [axes]
    

    while cap.isOpened():
        # Get frame and its tensor representation
        frame, frame_tensor = get_webcam_frame()
        if frame is None:
            break

        # Move tensor to the correct device
        frame_tensor = frame_tensor.to(device)
        
        # Run keypoint detection
        keypoints, scores_np = detect_keypoints(model, frame_tensor, args.threshold)

        # NEW: Filter out keypoints near the edges
        border_margin = 100  # Adjust this value to taste
        img_h, img_w = frame.shape[:2]
        filtered_keypoints = []
        for (x, y, score) in keypoints:
            if (x > border_margin and x < (img_w - border_margin) and
                y > border_margin and y < (img_h - border_margin)):
                filtered_keypoints.append((x, y, score))
        
        # Visualize keypoints on the frame
        blended, heatmap = visualize_keypoints(frame, keypoints, scores_np)
        
        # Display the results
        frame_disp = (frame * 255).astype(np.uint8)
        cv2.imshow('Original Frame', cv2.cvtColor(frame_disp, cv2.COLOR_RGB2BGR))
        cv2.imshow('Keypoints', cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

#    for i, image_file in enumerate(image_files):
#        image_path = os.path.join(args.image_dir, image_file)
#        print(f"Processing image {i+1}/{len(image_files)}: {image_path}")
#        
#        # Load image
#        img, img_tensor = get_webcam_frame(0)
#        img_tensor = img_tensor.to(device)
        
#        # Detect keypoints
#        keypoints, scores_np = detect_keypoints(model, img_tensor, args.threshold)
#        print(f"Detected {len(keypoints)} keypoints")
        
        # Visualize keypoints
#        blended, heatmap = visualize_keypoints(img, keypoints, scores_np)
        
#        # Save individual results
#        cv2.imwrite(os.path.join(args.output_dir, f"{image_file.split('.')[0]}_keypoints.png"), 
#                   cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
#        cv2.imwrite(os.path.join(args.output_dir, f"{image_file.split('.')[0]}_heatmap.png"), 
#                   heatmap)
        
        # Plot results
#        axes[i][0].imshow(img)
#        axes[i][0].set_title("Original Image")
#        axes[i][0].axis('off')
        
#        axes[i][1].imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
#        axes[i][1].set_title("Score Heatmap")
#        axes[i][1].axis('off')
        
#        axes[i][2].imshow(blended)
#        axes[i][2].set_title(f"Keypoints ({len(keypoints)} detected)")
#        axes[i][2].axis('off')
    
    # Save combined figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "keypoint_detection_results.png"))
    print(f"Results saved to {args.output_dir}")
    
    # Also create comparison with teacher model if available
    try:
        from models.teacher_model import TeacherModel
        teacher_path = "SuperPointPretrainedNetwork/superpoint_v1.pth"
        if os.path.exists(teacher_path):
            print("\nComparing with teacher model...")
            teacher = TeacherModel(
                superpoint_weights_path=teacher_path,
                num_clusters=64,
                backbone_out_channels=512
            ).to(device)
            teacher.eval()
            
            # Test on first image
            img, img_tensor = load_image(os.path.join(args.image_dir, image_files[0]))
            img_tensor = img_tensor.to(device)
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_scores, _, _ = teacher(img_tensor)
            
            teacher_scores_np = teacher_scores.squeeze().cpu().numpy()
            student_scores_np = scores_np
            
            # Visualize comparison
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.normalize(student_scores_np, None, 0, 1, cv2.NORM_MINMAX))
            plt.title("Student Model Score Map")
            plt.colorbar()
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.normalize(teacher_scores_np, None, 0, 1, cv2.NORM_MINMAX))
            plt.title("Teacher Model Score Map")
            plt.colorbar()
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, "teacher_student_comparison.png"))
    except Exception as e:
        print(f"Could not compare with teacher model: {e}")

if __name__ == "__main__":
    main()
