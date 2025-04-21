import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import os
import argparse
from glob import glob
from tqdm import tqdm
import sys
import argparse

# --- Add your project directory to Python path ---
# Adjust this path if necessary based on where you place this script
project_dir = os.path.dirname(os.path.abspath(__file__)) # Assumes script is in imwm_project
# If the script is elsewhere, provide the correct path:
# project_dir = '/path/to/your/Downloads/imwm_project' 
sys.path.insert(0, project_dir)
# ---------------------------------------------

try:
    from models.uwnet import UWNet
except ImportError:
    print("Error: Could not import UWNet model.")
    print("Make sure 'evaluate_hpatches.py' is placed correctly relative")
    print("to the 'models' directory or adjust the sys.path insertion above.")
    exit()

try:
    # Import the dense version needed for evaluation/distillation comparison
    from models.superpoint_pytorch import SuperPointDense
    # If SuperPointFrontend is still needed for comparison purposes,
    # you might need to define or import it separately, or adapt.
    # For *testing* the SuperPoint baseline, we'll use its output directly.
except ImportError as e:
    print(f"Error importing SuperPointDense: {e}")
    print("Make sure superpoint_pytorch.py is in the Python path.")
    exit()

# --- Helper Functions ---

def load_hpatches_homography(filepath):
    """Loads a 3x3 homography matrix from a text file."""
    try:
        return np.loadtxt(filepath, dtype=np.float64)
    except Exception as e:
        print(f"Error loading homography {filepath}: {e}")
        return None

def load_hpatches_image(image_path):
    """Load an image (PPM) for HPatches, convert to grayscale tensor."""
    try:
        # Use OpenCV which handles PPM well
        img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise IOError(f"Could not read image: {image_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # Normalize to [0, 1] for tensor conversion
        img_norm = img_gray.astype(np.float32) / 255.0
        # Add channel dimension and batch dimension: [1, 1, H, W]
        img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
        return img_rgb, img_gray, img_tensor # Return RGB for shape info, gray for kp extraction
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None, None, None


def extract_features(model, img_tensor, device, threshold=0.015, nms_kernel_size=5):
    """
    Run inference and extract keypoints (cv2.KeyPoint) and descriptors (np.array),
    including Non-Maximal Suppression (NMS). # Added NMS mention

    Args:
        model: The neural network model (e.g., UWNet).
        img_tensor: Input image tensor [1, C, H, W].
        device: The device ('cuda' or 'cpu').
        threshold (float): Keypoint detection confidence threshold.
        nms_kernel_size (int): Size of the kernel for NMS max pooling.

    Returns:
        keypoints (list): List of cv2.KeyPoint objects.
        descriptors (np.ndarray): NxD numpy array of descriptors, or None.
    """
    model.eval() # Ensure model is in eval mode
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        scores, dense_descriptors, _ = model(img_tensor) # Assume this output format

    # --- Non-Maximal Suppression (NMS) ---
    # Apply NMS on the raw scores tensor on the GPU for efficiency
    if scores.shape[1] != 1:
         print(f"Warning: Expected scores shape [N,1,H',W'], got {scores.shape}. Using first channel.")
         scores = scores[:, :1, :, :] # Take the first channel if multiple exist

    nms_padding = nms_kernel_size // 2
    # Max pooling finds the local maxima in a neighborhood
    local_max = F.max_pool2d(scores, kernel_size=nms_kernel_size, stride=1, padding=nms_padding)
    # Keep scores only where the original score equals the local maximum
    is_local_max = (scores == local_max)

    # Combine NMS mask with the threshold check
    # Points must be local maxima AND above the threshold
    keep_map = is_local_max & (scores > threshold)

    # Get coordinates of points to keep
    # nonzero() returns a tuple of tensors (batch_indices, channel_indices, y_indices, x_indices)
    keep_coords_tuple = keep_map.nonzero(as_tuple=True)

    if keep_coords_tuple[0].numel() == 0: # Check if any keypoints were found
        return [], None # No keypoints found after NMS and threshold

    # Extract the y and x coordinates (assuming batch size 1, channel 1)
    coords_y_tensor = keep_coords_tuple[2]
    coords_x_tensor = keep_coords_tuple[3]

    # Get the scores for the kept keypoints
    scores_filtered_tensor = scores[0, 0, coords_y_tensor, coords_x_tensor]

    # Convert final coordinates and scores to NumPy
    coords_y = coords_y_tensor.cpu().numpy()
    coords_x = coords_x_tensor.cpu().numpy()
    scores_values = scores_filtered_tensor.cpu().numpy()
    # --- End NMS ---


    # Scale coordinates back to original image size
    img_h, img_w = img_tensor.shape[2:] # Original image shape H, W
    score_h, score_w = scores.shape[2:] # Score map shape H', W' (use tensor shape)
    scale_x = img_w / score_w
    scale_y = img_h / score_h

    keypoints = []
    # Iterate using the NMS'd coordinates
    for i in range(len(coords_y)):
        # Use precise fractional coordinates from score map for warping
        # Center of the pixel in the score map
        raw_x = coords_x[i] + 0.5
        raw_y = coords_y[i] + 0.5
        # Scaled coordinates in original image
        img_x = raw_x * scale_x
        img_y = raw_y * scale_y
        score = scores_values[i]

        # Create cv2.KeyPoint object
        kp = cv2.KeyPoint(x=img_x, y=img_y, size=1, response=score)
        keypoints.append(kp)

    # --- Descriptor Extraction ---
    # Sample descriptors from the dense map at the NMS'd keypoint locations
    # IMPORTANT: Use the NMS'd coordinates (coords_y, coords_x) from the *score map*
    # dense_descriptors shape: (1, D, H', W')
    desc_dim = dense_descriptors.shape[1]
    # Ensure coordinates are valid indices for the descriptor map (should be if from score map)
    try:
        # Use the tensor coordinates before converting to numpy for direct indexing
        descriptors_sampled = dense_descriptors[0, :, coords_y_tensor, coords_x_tensor] # Shape: (D, N)
        descriptors_np = descriptors_sampled.T.cpu().numpy() # Shape: (N, D)
    except IndexError as e:
         print(f"Error indexing descriptors: {e}")
         print(f"Descriptor shape: {dense_descriptors.shape}, Max Y index: {coords_y_tensor.max()}, Max X index: {coords_x_tensor.max()}")
         return keypoints, None # Return keypoints found so far, but no descriptors


    # Optional: Limit number of keypoints (after NMS, before returning)
    n_features_limit = 2000 # Set your limit
    if len(keypoints) > n_features_limit:
         # Sort by response (score) and take top N
         indices = np.argsort(scores_values)[::-1][:n_features_limit]
         keypoints = [keypoints[i] for i in indices]
         descriptors_np = descriptors_np[indices]

    return keypoints, descriptors_np

def compute_repeatability(kp1, kp2, H_1_2, img_shape1, img_shape2, dist_thresh=3):
    """
    Calculates detector repeatability.

    Args:
        kp1 (list): Keypoints (cv2.KeyPoint) from image 1.
        kp2 (list): Keypoints (cv2.KeyPoint) from image 2.
        H_1_2 (np.ndarray): Homography transforming points from img1 to img2.
        img_shape1 (tuple): Shape (h, w) of image 1.
        img_shape2 (tuple): Shape (h, w) of image 2.
        dist_thresh (int): Pixel distance threshold for correspondence.

    Returns:
        float: Repeatability score, or np.nan if calculation is not possible.
    """
    if not kp1 or not kp2 or H_1_2 is None:
        return np.nan

    h1, w1 = img_shape1[:2]
    h2, w2 = img_shape2[:2]

    # Define corners of image 1
    corners1 = np.array([
        [0, 0], [w1 - 1, 0], [w1 - 1, h1 - 1], [0, h1 - 1]
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to image 2 coordinate frame
    corners1_warped = cv2.perspectiveTransform(corners1, H_1_2)

    # Define bounding box of warped image 1 in image 2 frame
    min_x_warp, min_y_warp = np.min(corners1_warped.squeeze(), axis=0)
    max_x_warp, max_y_warp = np.max(corners1_warped.squeeze(), axis=0)

    # Define bounding box of image 2
    min_x2, min_y2 = 0, 0
    max_x2, max_y2 = w2 - 1, h2 - 1

    # Find the intersection (overlap) bounding box
    min_x_overlap = max(min_x_warp, min_x2)
    min_y_overlap = max(min_y_warp, min_y2)
    max_x_overlap = min(max_x_warp, max_x2)
    max_y_overlap = min(max_y_warp, max_y2)

    # Get keypoint coordinates as numpy arrays
    pts1 = np.array([p.pt for p in kp1], dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array([p.pt for p in kp2], dtype=np.float32) # Shape (N2, 2)

    # Warp points from image 1 to image 2
    pts1_warped = cv2.perspectiveTransform(pts1, H_1_2) # Shape (N1, 1, 2)
    pts1_warped = pts1_warped.squeeze(1) # Shape (N1, 2)

    # Filter points that fall within the overlap region
    valid_warp_indices = np.where(
        (pts1_warped[:, 0] >= min_x_overlap) & (pts1_warped[:, 0] <= max_x_overlap) &
        (pts1_warped[:, 1] >= min_y_overlap) & (pts1_warped[:, 1] <= max_y_overlap)
    )[0]

    valid_kp2_indices = np.where(
        (pts2[:, 0] >= min_x_overlap) & (pts2[:, 0] <= max_x_overlap) &
        (pts2[:, 1] >= min_y_overlap) & (pts2[:, 1] <= max_y_overlap)
    )[0]

    pts1_warped_overlap = pts1_warped[valid_warp_indices]
    pts2_overlap = pts2[valid_kp2_indices]

    num_kp1_in_overlap = len(pts1_warped_overlap)
    if num_kp1_in_overlap == 0 or len(pts2_overlap) == 0:
        return 0.0 if num_kp1_in_overlap == 0 else np.nan # Avoid division by zero if no kp1 in overlap

    # Find correspondences using distance threshold
    correspondences = 0
    # Use a KD-Tree for faster nearest neighbor search if many points
    # from scipy.spatial import cKDTree
    # tree = cKDTree(pts2_overlap)
    # distances, indices = tree.query(pts1_warped_overlap, k=1, distance_upper_bound=dist_thresh)
    # correspondences = np.sum(distances <= dist_thresh)

    # Simple brute-force check (okay for moderate number of keypoints)
    for pt1_w in pts1_warped_overlap:
        distances = np.linalg.norm(pts2_overlap - pt1_w, axis=1)
        if np.min(distances) <= dist_thresh:
            correspondences += 1

    repeatability = correspondences / num_kp1_in_overlap
    return repeatability

def compute_ms(kp1, desc1, kp2, desc2, H_1_2, dist_thresh_gt=3, ratio_thresh=0.75, norm_type=cv2.NORM_L2):
    """
    Calculates Matching Score (%M.S.).

    The Matching Score is the ratio of ground truth correspondences
    that are correctly found by a putative matcher (e.g., BFMatcher + Ratio Test).

    Args:
        kp1 (list): Keypoints (cv2.KeyPoint) from image 1.
        desc1 (np.ndarray): Descriptors (N1xD) from image 1.
        kp2 (list): Keypoints (cv2.KeyPoint) from image 2.
        desc2 (np.ndarray): Descriptors (N2xD) from image 2.
        H_1_2 (np.ndarray): Homography transforming points from img1 to img2.
        dist_thresh_gt (int): Pixel distance threshold for ground truth correspondence.
        ratio_thresh (float): Lowe's ratio test threshold for putative matching.
        norm_type: OpenCV norm type for BFMatcher (e.g., cv2.NORM_L2, cv2.NORM_HAMMING).

    Returns:
        float: Matching Score, or np.nan if calculation not possible.
    """
    # --- Input validation ---
    if desc1 is None or desc2 is None or not kp1 or not kp2 or H_1_2 is None:
        return np.nan
    if desc1.shape[0] != len(kp1) or desc2.shape[0] != len(kp2):
         print(f"Warning (MS): Mismatch between keypoints and descriptors count!")
         return np.nan
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return 0.0 # No keypoints to potentially match

    # --- 1. Find Ground Truth Correspondences ---
    # (This part is identical to the beginning of compute_mma)
    pts1 = np.array([p.pt for p in kp1], dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array([p.pt for p in kp2], dtype=np.float32).reshape(-1, 1, 2)
    pts1_warped = cv2.perspectiveTransform(pts1, H_1_2) # Shape (N1, 1, 2)

    kp1_indices_with_gt = set() # Set of kp1 indices that have a GT match in kp2
    # Brute-force nearest neighbor in coordinate space for GT
    for i in range(pts1_warped.shape[0]):
        pt1_w = pts1_warped[i] # Shape (1, 2)
        distances = np.linalg.norm(pts2 - pt1_w, axis=2).flatten() # Distances to all kp2
        best_match_idx = np.argmin(distances)
        if distances[best_match_idx] < dist_thresh_gt:
            kp1_indices_with_gt.add(i)

    num_gt_pairs = len(kp1_indices_with_gt)
    if num_gt_pairs == 0:
        # If there are no ground truth pairs possible, the score is undefined or trivially 0/0.
        # Return 0.0, as no GT points could be potentially matched.
        return 0.0

    # --- 2. Perform Putative Matching (BFMatcher + Ratio Test) ---
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    # Find 2 nearest neighbors for each descriptor in desc1
    matches_knn = bf.knnMatch(desc1, desc2, k=2)

    # Apply Lowe's ratio test and collect indices of kp1 that passed
    kp1_indices_matched_putatively = set()
    for match_pair in matches_knn:
        # Ensure we have two neighbors to compare
        if len(match_pair) == 2:
            m, n = match_pair
            # Apply Lowe's ratio test
            if m.distance < ratio_thresh * n.distance:
                # m.queryIdx is the index in kp1/desc1
                kp1_indices_matched_putatively.add(m.queryIdx)

    # --- 3. Calculate Matching Score ---
    # Find the intersection: ground truth keypoints (in kp1) that were
    # also successfully matched by the putative matcher.
    correctly_found_gt_indices = kp1_indices_with_gt.intersection(kp1_indices_matched_putatively)
    num_gt_putatively_matched = len(correctly_found_gt_indices)

    # M.S. = (Number of GT points found by putative matcher) / (Total number of GT points)
    ms_score = num_gt_putatively_matched / num_gt_pairs

    return ms_score

def compute_mma(kp1, desc1, kp2, desc2, H_1_2, dist_thresh=3, norm_type=cv2.NORM_L2):
    """
    Calculates Mean Matching Accuracy (MMA).

    Args:
        kp1 (list): Keypoints (cv2.KeyPoint) from image 1.
        desc1 (np.ndarray): Descriptors (N1xD) from image 1.
        kp2 (list): Keypoints (cv2.KeyPoint) from image 2.
        desc2 (np.ndarray): Descriptors (N2xD) from image 2.
        H_1_2 (np.ndarray): Homography transforming points from img1 to img2.
        dist_thresh (int): Pixel distance threshold for ground truth correspondence.
        norm_type: OpenCV norm type for BFMatcher (e.g., cv2.NORM_L2, cv2.NORM_HAMMING).

    Returns:
        float: MMA score, or np.nan if calculation not possible.
    """
    if desc1 is None or desc2 is None or not kp1 or not kp2 or H_1_2 is None:
        return np.nan

    if desc1.shape[0] != len(kp1) or desc2.shape[0] != len(kp2):
         print(f"Warning: Mismatch between keypoints and descriptors count! ({len(kp1)} vs {desc1.shape[0]}, {len(kp2)} vs {desc2.shape[0]})")
         return np.nan # Mismatch indicates an error upstream

    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return 0.0 # No keypoints/descriptors to match

    # 1. Find Ground Truth Correspondences using Homography
    pts1 = np.array([p.pt for p in kp1], dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array([p.pt for p in kp2], dtype=np.float32).reshape(-1, 1, 2)
    pts1_warped = cv2.perspectiveTransform(pts1, H_1_2) # Shape (N1, 1, 2)

    gt_matches = {} # Dictionary: kp1_index -> kp2_index
    kp1_indices_with_gt = []
    # Brute-force nearest neighbor in coordinate space for GT
    for i in range(pts1_warped.shape[0]):
        pt1_w = pts1_warped[i] # Shape (1, 2)
        distances = np.linalg.norm(pts2 - pt1_w, axis=2).flatten() # Distances to all kp2
        best_match_idx = np.argmin(distances)
        if distances[best_match_idx] < dist_thresh:
            gt_matches[i] = best_match_idx
            kp1_indices_with_gt.append(i)

    num_gt_pairs = len(kp1_indices_with_gt)
    if num_gt_pairs == 0:
        return 0.0 # No ground truth pairs found within threshold

    # 2. Match Descriptors using Nearest Neighbor
    bf = cv2.BFMatcher(norm_type, crossCheck=False) # Use crossCheck=False for MMA
    # We only need to match descriptors from kp1 that have a ground truth correspondence
    if not kp1_indices_with_gt:
        return 0.0
    desc1_subset = desc1[kp1_indices_with_gt]
    # Query: desc1_subset, Train: desc2
    matches_knn = bf.knnMatch(desc1_subset, desc2, k=1) # Find 1 NN for each desc1_subset

    # 3. Check Correctness of Descriptor Matches
    correct_matches = 0
    potential_matches = 0 # Should equal num_gt_pairs if knnMatch finds a match for all

    for i, m_list in enumerate(matches_knn):
        if not m_list: continue # Skip if no match found for this desc1_subset descriptor
        potential_matches += 1
        m = m_list[0] # Get the single nearest neighbor match object

        query_idx_subset = m.queryIdx # Index within the *subset* desc1_subset
        train_idx = m.trainIdx       # Index within the *full* desc2

        # Map query_idx_subset back to the original index in kp1/desc1
        original_query_idx = kp1_indices_with_gt[query_idx_subset]

        # Check if this descriptor match (original_query_idx -> train_idx)
        # corresponds to a ground truth geometric match
        if original_query_idx in gt_matches and gt_matches[original_query_idx] == train_idx:
            correct_matches += 1

    # MMA calculation
    if potential_matches == 0:
         # This case might happen if knnMatch returns empty lists for all queries
        return 0.0

    mma = correct_matches / potential_matches # Or sometimes num_gt_pairs, check benchmarks
    # Using potential_matches as denominator: accuracy among those features for which a NN was found.
    # Using num_gt_pairs as denominator: accuracy among all features that *should* have a match. Standard MMA often uses num_gt_pairs.
    mma_alt = correct_matches / num_gt_pairs
    # Let's stick to the definition using num_gt_pairs for consistency with common practice.
    mma = correct_matches / num_gt_pairs

    return mma

# --- NEW: Extractor function for SuperPoint using SuperPointDense ---
def extract_superpoint_dense_features(model, img_tensor, device, threshold=0.015, nms_kernel_size=5):
    """
    Extract features using the SuperPointDense model.
    Applies NMS and thresholding to the dense score map.

    Args:
        model (SuperPointDense): The loaded SuperPointDense model.
        img_tensor (torch.Tensor): Input image tensor [1, C, H, W].
        device (str): 'cuda' or 'cpu'.
        threshold (float): Keypoint detection confidence threshold.
        nms_kernel_size (int): Size of the kernel for NMS max pooling.


    Returns:
        list: List of cv2.KeyPoint objects.
        np.ndarray: NxD numpy array of descriptors (N x 256), float32.
    """
    model.eval() # Ensure model is in eval mode
    img_tensor = img_tensor.to(device)

    # Ensure input is grayscale for SuperPointDense
    if img_tensor.shape[1] == 3:
        scale = img_tensor.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        img_tensor_gray = (img_tensor * scale).sum(1, keepdim=True)
    else:
        img_tensor_gray = img_tensor

    with torch.no_grad():
        outputs = model(img_tensor_gray) # Get {'dense_score': ..., 'dense_desc': ...}
        scores = outputs['dense_score'] # (N, 1, H/8, W/8)
        dense_descriptors = outputs['dense_desc'] # (N, 256, H/8, W/8)

    # --- Apply NMS (Same logic as used for UWNet) ---
    nms_padding = nms_kernel_size // 2
    local_max = F.max_pool2d(scores, kernel_size=nms_kernel_size, stride=1, padding=nms_padding)

    # --- CROPPING ---
    # Crop the local_max tensor to match the original scores tensor size
    # This handles potential size mismatches caused by padding in max_pool2d
    _, _, H_scores, W_scores = scores.shape
    local_max = local_max[:, :, :H_scores, :W_scores] # Crop height and width
    # --- END CROPPING ---

    is_local_max = (scores == local_max)
    keep_map = is_local_max & (scores > threshold)
    keep_coords_tuple = keep_map.nonzero(as_tuple=True)

    if keep_coords_tuple[0].numel() == 0:
        return [], None

    coords_y_tensor = keep_coords_tuple[2]
    coords_x_tensor = keep_coords_tuple[3]
    scores_filtered_tensor = scores[0, 0, coords_y_tensor, coords_x_tensor]

    coords_y = coords_y_tensor.cpu().numpy()
    coords_x = coords_x_tensor.cpu().numpy()
    scores_values = scores_filtered_tensor.cpu().numpy()
    # --- End NMS ---

    # Scale coordinates (Same logic as UWNet)
    img_h, img_w = img_tensor.shape[2:] # Use original tensor shape
    score_h, score_w = scores.shape[2:]
    scale_x = img_w / score_w
    scale_y = img_h / score_h

    keypoints = []
    for i in range(len(coords_y)):
        raw_x = coords_x[i] + 0.5
        raw_y = coords_y[i] + 0.5
        img_x = raw_x * scale_x
        img_y = raw_y * scale_y
        score = scores_values[i]
        kp = cv2.KeyPoint(x=float(img_x), y=float(img_y), size=1, response=float(score))
        keypoints.append(kp)

    # Descriptor Extraction (Sample from dense map at NMS'd coords)
    desc_dim = dense_descriptors.shape[1]
    descriptors_np = np.zeros((0, desc_dim), dtype=np.float32) # Default empty
    try:
        if len(keypoints) > 0:
            descriptors_sampled = dense_descriptors[0, :, coords_y_tensor, coords_x_tensor]
            descriptors_np = descriptors_sampled.T.cpu().numpy().astype(np.float32)
    except IndexError as e:
         print(f"Error indexing descriptors: {e}")
         return keypoints, None

    # Optional: Limit features
    n_features_limit = 2000
    if len(keypoints) > n_features_limit:
        indices = np.argsort(scores_values)[::-1][:n_features_limit]
        keypoints = [keypoints[i] for i in indices]
        descriptors_np = descriptors_np[indices]


    return keypoints, descriptors_np
    
# Example: Inside a new function or modified extract_features
def extract_sift_features(img_gray, n_features=2000): # Limit features for comparison
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    # Ensure descriptors are float32 if None (sometimes happens if no kpts found)
    if descriptors is None:
         descriptors = np.array([], dtype=np.float32).reshape(0, 128)
    return keypoints, descriptors.astype(np.float32) # Ensure float32

# Similarly for ORB:
def extract_orb_features(img_gray, n_features=2000):
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)
    # ORB descriptors are uint8
    if descriptors is None:
        descriptors = np.array([], dtype=np.uint8).reshape(0, 32)
    return keypoints, descriptors

# --- Main Evaluation Function ---

def main():
    parser = argparse.ArgumentParser(description='Evaluate UWNet on HPatches')
    parser.add_argument('--method', type=str, required=True,
                        choices=['uwnet', 'sift', 'orb', 'superpoint'], # Add more as you implement
                        help='Feature extraction method to evaluate.')
    parser.add_argument('hpatches_dir', type=str,
                        help='Path to the root HPatches directory (e.g., hpatches-sequences-release)')
    parser.add_argument('--model_path', type=str, default='models/UWNet_v7.pth', # Updated default model name
                        help='Path to trained UWNet model')
    parser.add_argument('--threshold', type=float, default=0.015,
                        help='Keypoint detection threshold')
    parser.add_argument('--n_features', type=int, default=2000, # Parameter for SIFT/ORB
                        help='Max number of features for SIFT/ORB')
    parser.add_argument('--sp_weights_path', type=str, default='SuperPointPretrainedNetwork/superpoint_v1.pth', # Default path
                        help='Path to SuperPoint pretrained weights')
    parser.add_argument('--sp_nms_dist', type=int, default=4,
                        help='SuperPoint NMS distance.')
    parser.add_argument('--rep_thresh', type=int, default=3,
                        help='Repeatability distance threshold (pixels)')
    parser.add_argument('--mma_thresh', type=int, default=3,
                        help='MMA/MS ground truth distance threshold (pixels)')
    parser.add_argument('--ratio_thresh', type=float, default=0.75,
                        help='Lowe\'s ratio test threshold for Matching Score')
    parser.add_argument('--sequences', type=str, default=None,
                        help='Comma-separated list of specific sequences to run (e.g., v_boat,i_ajuntament)')
    args = parser.parse_args()

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Evaluating method: {args.method}")

    # --- Load/Initialize selected method ---
    model = None # For deep models
    extractor_func = None
    metric_norm_type = None

    if args.method == 'uwnet':
        print(f"Loading UWNet model from {args.model_path}")
        model = UWNet().to(device)
        try:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.eval()
            print("UWNet Model loaded successfully.")
            # Define a wrapper to match the expected signature
            def uwnet_extractor(img_tensor, device, threshold):
                return extract_features(model, img_tensor, device, threshold) # Your original function
            extractor_func = uwnet_extractor
            metric_norm_type = cv2.NORM_L2
        except Exception as e:
            print(f"Error loading UWNet model: {e}")
            return
    elif args.method == 'sift':
        print(f"Using OpenCV SIFT (nfeatures={args.n_features})")
        # Define a wrapper to match expected signature (takes tensor, device, threshold)
        def sift_extractor(img_tensor, device, threshold):
            # Convert tensor back to numpy grayscale uint8 for OpenCV
            img_gray_np = img_tensor.squeeze().cpu().numpy() # Assuming N=1, C=1
            img_gray_u8 = (img_gray_np * 255).astype(np.uint8)
            return extract_sift_features(img_gray_u8, n_features=args.n_features)
        extractor_func = sift_extractor
        metric_norm_type = cv2.NORM_L2 # SIFT uses L2
    elif args.method == 'orb':
        print(f"Using OpenCV ORB (nfeatures={args.n_features})")
        def orb_extractor(img_tensor, device, threshold):
             img_gray_np = img_tensor.squeeze().cpu().numpy()
             img_gray_u8 = (img_gray_np * 255).astype(np.uint8)
             return extract_orb_features(img_gray_u8, n_features=args.n_features)
        extractor_func = orb_extractor
        metric_norm_type = cv2.NORM_HAMMING # ORB uses Hamming
    elif args.method == 'superpoint':
        print(f"Loading SuperPointDense model with weights: {args.sp_weights_path}")
        try:
            # Load the SuperPointDense model
            sp_model = SuperPointDense().to(device)
            sp_model.load_state_dict(torch.load(args.sp_weights_path, map_location=device))
            sp_model.eval()
            print("SuperPointDense model loaded successfully.")

            # Define the extractor function using the loaded model
            # Use a lambda to capture the sp_model instance
            # Ensure extract_superpoint_dense_features is defined above main()
            extractor_func = lambda img_tensor, device, threshold: extract_superpoint_dense_features(sp_model, img_tensor, device, threshold, args.sp_nms_dist) # Pass NMS dist

            metric_norm_type = cv2.NORM_L2 # SuperPoint descriptors are float, use L2

        except FileNotFoundError:
            print(f"ERROR: SuperPoint weights file not found at {args.sp_weights_path}")
            print("Please check the path provided via --sp_weights_path")
            return
        except Exception as e:
            print(f"Error loading SuperPointDense model: {e}")
            # Print detailed traceback
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"Error: Method {args.method} not recognized or implemented.")
        return

    # Find sequences
    sequence_dirs = sorted(glob(os.path.join(args.hpatches_dir, '*')))
    sequence_dirs = [d for d in sequence_dirs if os.path.isdir(d)]
    if not sequence_dirs:
        print(f"Error: No sequences found in {args.hpatches_dir}")
        return

    # Filter sequences if requested
    if args.sequences:
        selected_sequences = args.sequences.split(',')
        sequence_dirs = [d for d in sequence_dirs if os.path.basename(d) in selected_sequences]
        if not sequence_dirs:
             print(f"Error: None of the specified sequences found: {args.sequences}")
             return
        print(f"Running evaluation on specific sequences: {selected_sequences}")

    print(f"Found {len(sequence_dirs)} sequences to evaluate.")

    # --- Initialize results dictionary including 'ms' ---
    results = {'illumination': {'repeatability': [], 'mma': [], 'ms': []},
               'viewpoint':    {'repeatability': [], 'mma': [], 'ms': []}}
    # ---------------------------------------------------

    # --- Evaluation Loop ---
    for seq_dir in tqdm(sequence_dirs, desc="Evaluating Sequences"):
        seq_name = os.path.basename(seq_dir)
        seq_type = 'illumination' if seq_name.startswith('i_') else 'viewpoint'

        # Load reference image (1.ppm)
        ref_img_path = os.path.join(seq_dir, '1.ppm')
        img1_rgb, img1_gray, img1_tensor = load_hpatches_image(ref_img_path)
        if img1_tensor is None:
            print(f"Skipping sequence {seq_name} due to image loading error.")
            continue

        # --- Extract features using the selected method's function ---
        kp1, desc1 = extractor_func(img1_tensor, device, args.threshold)
        # -----------------------------------------------------------

        # Loop through target images (2.ppm to 6.ppm)
        for i in range(2, 7):
            target_img_path = os.path.join(seq_dir, f'{i}.ppm')
            homography_path = os.path.join(seq_dir, f'H_1_{i}')

            img_i_rgb, img_i_gray, img_i_tensor = load_hpatches_image(target_img_path)
            H_1_i = load_hpatches_homography(homography_path)

            if img_i_tensor is None or H_1_i is None:
                print(f"Skipping pair (1, {i}) in sequence {seq_name} due to loading error.")
                continue

            # --- Extract target features ---
            kp_i, desc_i = extractor_func(img_i_tensor, device, args.threshold)
            # -----------------------------

            # --- Calculate ALL metrics ---
            rep = compute_repeatability(kp1, kp_i, H_1_i, img1_gray.shape, img_i_gray.shape, args.rep_thresh)

            # metric_norm_type is set correctly based on method above
            mma = compute_mma(kp1, desc1, kp_i, desc_i, H_1_i, args.mma_thresh, norm_type=metric_norm_type)
            ms = compute_ms(kp1, desc1, kp_i, desc_i, H_1_i, args.mma_thresh, args.ratio_thresh, norm_type=metric_norm_type)
            # ----------------------------

            # --- Store results (handle potential NaNs) ---
            if not np.isnan(rep):
                results[seq_type]['repeatability'].append(rep)
            if not np.isnan(mma):
                results[seq_type]['mma'].append(mma)
            if not np.isnan(ms): # Store ms result
                results[seq_type]['ms'].append(ms)
            # ----------------------------------------------

    # --- Report Results ---
    print("\n--- Evaluation Results ---")

    for seq_type in ['illumination', 'viewpoint']:
        print(f"\n{seq_type.capitalize()} Sequences:")
        rep_scores = results[seq_type]['repeatability']
        mma_scores = results[seq_type]['mma']
        ms_scores = results[seq_type]['ms'] # Get ms scores list

        # Print Repeatability
        if rep_scores:
            avg_rep = np.mean(rep_scores)
            print(f"  Average Repeatability: {avg_rep:.4f} ({len(rep_scores)} pairs)")
        else:
            print("  Average Repeatability: N/A (No valid pairs)")

        # Print MMA
        if mma_scores:
            avg_mma = np.mean(mma_scores)
            print(f"  Average MMA:           {avg_mma:.4f} ({len(mma_scores)} pairs)")
        else:
            print("  Average MMA:           N/A (No valid pairs)")

        # --- Print Matching Score ---
        if ms_scores:
            avg_ms = np.mean(ms_scores)
            print(f"  Average M.S.:          {avg_ms:.4f} ({len(ms_scores)} pairs)") # Print avg_ms
        else:
            print("  Average M.S.:          N/A (No valid pairs)")
        # ---------------------------

    print("\nEvaluation finished.")


if __name__ == "__main__":
    # Ensure all helper functions (load_*, extract_*, compute_*) are defined above main()
    main()