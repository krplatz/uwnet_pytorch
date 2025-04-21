#!/usr/bin/env python
# UWNet Keypoint Tracker (using UWNet's own score map + Fundamental/Homography RANSAC)

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from models.uwnet import UWNet
from PIL import Image
import os
import argparse
import time

# If you still need PointTracker, you can import it:
# from SuperPointPretrainedNetwork.demo_superpoint import PointTracker


def load_image(image_path):
    """Load an image and convert to tensor [0,1]. Also return float32 np array for grayscale if needed."""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Tensor shape: (1, 3, H, W)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
    return img, gray, img_tensor


def detect_keypoints_and_descriptors_with_scores(
    model,
    img_tensor,
    score_thresh=0.015,
    max_keypoints=500
):
    """
    1) Run UWNet => get scores + descriptors.
    2) Threshold scores to pick keypoints.
    3) Interpolate descriptors at keypoint locations.
    4) Return (pts, desc, score_map_for_vis).
       - pts shape: (3, N), where row0=x, row1=y, row2=score
       - desc shape: (dim, N)
    """
    device = img_tensor.device
    # 1. Forward pass
    with torch.no_grad():
        scores, descriptors, _ = model(img_tensor)

    # Convert to CPU numpy
    scores_np = scores.squeeze().cpu().numpy()            # shape: (Hc, Wc)
    descriptors_np = descriptors.squeeze().cpu().numpy()  # shape: (dim, Hc, Wc)

    Hc, Wc = scores_np.shape
    desc_dim = descriptors_np.shape[0]  # e.g. 256
    img_h, img_w = img_tensor.shape[2:]  # original image size

    # 2. Find all points above threshold
    coords = np.where(scores_np > score_thresh)
    scores_filtered = scores_np[coords]

    if len(scores_filtered) == 0:
        # No keypoints found
        return np.zeros((3, 0)), np.zeros((desc_dim, 0)), np.zeros((img_h, img_w), dtype=np.float32)

    # Sort by descending score
    sort_idx = np.argsort(scores_filtered)[::-1]
    coords_y = coords[0][sort_idx]
    coords_x = coords[1][sort_idx]
    scores_sorted = scores_filtered[sort_idx]

    # 3. Keep only top max_keypoints
    if len(scores_sorted) > max_keypoints:
        coords_y = coords_y[:max_keypoints]
        coords_x = coords_x[:max_keypoints]
        scores_sorted = scores_sorted[:max_keypoints]

    # 4. Convert from descriptor map coords -> full image coords
    scale_y = float(img_h) / float(Hc)
    scale_x = float(img_w) / float(Wc)

    # Build the final Nx3 array: [x, y, score]
    pts = np.zeros((3, len(coords_y)), dtype=np.float32)
    for i in range(len(coords_y)):
        y_c = coords_y[i]
        x_c = coords_x[i]
        s = scores_sorted[i]

        # scale up
        x_full = x_c * scale_x
        y_full = y_c * scale_y

        pts[0, i] = x_full
        pts[1, i] = y_full
        pts[2, i] = s

    # 5. Interpolate descriptors at these locations (nearest-neighbor for simplicity)
    desc_list = []
    for i in range(pts.shape[1]):
        x_full = pts[0, i]
        y_full = pts[1, i]

        # Convert to descriptor-map coordinates
        x_desc = x_full / scale_x
        y_desc = y_full / scale_y

        # Round or clamp
        x0 = int(round(x_desc))
        y0 = int(round(y_desc))
        x0 = max(0, min(x0, Wc - 1))
        y0 = max(0, min(y0, Hc - 1))

        desc_vec = descriptors_np[:, y0, x0]
        desc_list.append(desc_vec)

    descriptors_out = np.array(desc_list).T  # shape: (dim, N)

    # Normalize descriptors
    norms = np.linalg.norm(descriptors_out, axis=0, keepdims=True)
    norms[norms == 0] = 1e-10
    descriptors_out /= norms

    # 6. Create a "score map" for visualization (upsample to original image size)
    score_map_full = cv2.resize(scores_np, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
    if score_map_full.max() > 0:
        score_map_full /= score_map_full.max()

    return pts, descriptors_out, score_map_full


def draw_keypoints(image, keypoints, color=(0, 255, 0), radius=4):
    """Draw keypoints on an image (circle). keypoints shape: (3, N)."""
    image_with_kp = image.copy()
    for i in range(keypoints.shape[1]):
        x, y = int(round(keypoints[0, i])), int(round(keypoints[1, i]))
        cv2.circle(image_with_kp, (x, y), radius, color, -1)
        cv2.circle(image_with_kp, (x, y), radius, (0, 0, 0), 1)
    return image_with_kp


def draw_matches(img1, kp1, img2, kp2, matches, color=(0, 255, 0), thickness=2):
    """Draw matches between two images. matches shape: (3, N)."""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = img1
    out[:h2, w1:w1 + w2] = img2

    for i in range(matches.shape[1]):
        idx1, idx2 = int(matches[0, i]), int(matches[1, i])
        pt1 = (int(round(kp1[0, idx1])), int(round(kp1[1, idx1])))
        pt2 = (int(round(kp2[0, idx2])) + w1, int(round(kp2[1, idx2])))

        cv2.line(out, pt1, pt2, color, thickness)
        radius = 5
        cv2.circle(out, pt1, radius, color, -1)
        cv2.circle(out, pt2, radius, color, -1)
        cv2.circle(out, pt1, radius, (0, 0, 0), 1)
        cv2.circle(out, pt2, radius, (0, 0, 0), 1)

    return out


def nn_match_two_way_ratio(desc1, desc2, ratio_thresh=0.8):
    """Two-way nearest neighbor matching with ratio test, returns (3, N): idx1, idx2, distance."""
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
        return np.zeros((3, 0))

    # L2 distance via dot product trick
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2 - 2 * np.clip(dmat, -1, 1))

    # Sort each row to find best + second-best
    idx_sort = np.argsort(dmat, axis=1)
    best_idx = idx_sort[:, 0]
    second_idx = idx_sort[:, 1]

    best_scores = dmat[np.arange(dmat.shape[0]), best_idx]
    second_scores = dmat[np.arange(dmat.shape[0]), second_idx]

    ratio = best_scores / (second_scores + 1e-10)
    keep1 = ratio < ratio_thresh

    # mutual check
    idx_sort2 = np.argsort(dmat, axis=0)
    best_idx2 = idx_sort2[0, :]
    keep_matches = []
    for i in range(len(keep1)):
        if keep1[i]:
            j = best_idx[i]
            if j < best_idx2.shape[0] and best_idx2[j] == i:
                keep_matches.append(i)

    keep_matches = np.array(keep_matches, dtype=int)
    if len(keep_matches) == 0:
        return np.zeros((3, 0))

    m_idx1 = keep_matches
    m_idx2 = best_idx[m_idx1]
    m_scores = best_scores[m_idx1]

    matches = np.zeros((3, len(m_idx1)))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = m_scores

    return matches


def filter_matches_ransac(kp1, kp2, matches, ransac_thresh=5.0, method='fundamental'):
    """
    Filter matches using RANSAC with either a fundamental matrix or homography.
    kp1, kp2: (3, N)
    matches: (3, M)
    """
    if matches.shape[1] == 0:
        return matches

    src_pts = []
    dst_pts = []
    for i in range(matches.shape[1]):
        idx1 = int(matches[0, i])
        idx2 = int(matches[1, i])
        src_pts.append([kp1[0, idx1], kp1[1, idx1]])
        dst_pts.append([kp2[0, idx2], kp2[1, idx2]])

    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    if method == 'fundamental':
        # General 3D scene: fundamental matrix
        M, inliers = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, ransac_thresh, 0.99)
    elif method == 'homography':
        # Mostly planar scene: homography
        M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    else:
        raise ValueError(f"Unknown RANSAC method: {method}")

    if inliers is None:
        return np.zeros((3, 0))

    inliers = inliers.ravel().astype(bool)
    inlier_matches = matches[:, inliers]
    return inlier_matches


def main():
    parser = argparse.ArgumentParser(description='UWNet Keypoint Tracker (Fundamental/Homography RANSAC)')
    parser.add_argument('--model_path', type=str, default='models/UWNet_v9.pth',
                        help='Path to trained UWNet model')
    parser.add_argument('--image1', type=str, default='pseudo_images/NYU0035.png',
                        help='Path to first image')
    parser.add_argument('--image2', type=str, default='pseudo_images/NYU0036.png',
                        help='Path to second image')
    parser.add_argument('--max_keypoints', type=int, default=1000,
                        help='Maximum number of keypoints to detect (default: 1000)')
    parser.add_argument('--score_thresh', type=float, default=0.0015,
                        help='Threshold for UWNet score map (default: 0.0015)')
    parser.add_argument('--ratio_thresh', type=float, default=0.95,
                        help='Ratio test threshold for descriptor matching (default: 0.95)')
    parser.add_argument('--ransac_thresh', type=float, default=10.0,
                        help='RANSAC reprojection threshold (default: 10.0)')
    parser.add_argument('--ransac_method', type=str, default='fundamental',
                        help='RANSAC method: "fundamental" or "homography" (default: fundamental)')
    parser.add_argument('--output_dir', type=str, default='data_visualization/uwnet_tracking_results',
                        help='Directory to save results')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA GPU acceleration')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = UWNet().to(device)
    print(f"Loading model from {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process first image
    print(f"Processing first image: {args.image1}")
    img1, gray1, img1_tensor = load_image(args.image1)
    img1_tensor = img1_tensor.to(device)

    t0 = time.time()
    kp1, desc1, score_map1 = detect_keypoints_and_descriptors_with_scores(
        model,
        img1_tensor,
        score_thresh=args.score_thresh,
        max_keypoints=args.max_keypoints
    )
    t1 = time.time()
    print(f"Detected {kp1.shape[1]} keypoints in {t1 - t0:.3f} seconds")

    # Process second image
    print(f"Processing second image: {args.image2}")
    img2, gray2, img2_tensor = load_image(args.image2)
    img2_tensor = img2_tensor.to(device)

    t0 = time.time()
    kp2, desc2, score_map2 = detect_keypoints_and_descriptors_with_scores(
        model,
        img2_tensor,
        score_thresh=args.score_thresh,
        max_keypoints=args.max_keypoints
    )
    t1 = time.time()
    print(f"Detected {kp2.shape[1]} keypoints in {t1 - t0:.3f} seconds")

    # Now match descriptors
    print("Matching descriptors with ratio test...")
    matches = nn_match_two_way_ratio(desc1, desc2, ratio_thresh=args.ratio_thresh)
    print(f"Found {matches.shape[1]} raw matches (ratio test)")

    print(f"Applying {args.ransac_method} RANSAC to remove outliers...")
    matches_inliers = filter_matches_ransac(
        kp1, kp2, matches,
        ransac_thresh=args.ransac_thresh,
        method=args.ransac_method
    )
    print(f"Kept {matches_inliers.shape[1]} inlier matches after RANSAC")

    # Visualize
    img1_vis = (img1 * 255).astype(np.uint8)
    img2_vis = (img2 * 255).astype(np.uint8)

    img1_kp = draw_keypoints(cv2.cvtColor(img1_vis, cv2.COLOR_RGB2BGR), kp1)
    img2_kp = draw_keypoints(cv2.cvtColor(img2_vis, cv2.COLOR_RGB2BGR), kp2)

    # Convert score maps to color heatmaps
    sm1_u8 = cv2.normalize(score_map1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sm1_color = cv2.applyColorMap(sm1_u8, cv2.COLORMAP_JET)
    blend1 = cv2.addWeighted(img1_kp, 0.7, sm1_color, 0.3, 0)

    sm2_u8 = cv2.normalize(score_map2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sm2_color = cv2.applyColorMap(sm2_u8, cv2.COLORMAP_JET)
    blend2 = cv2.addWeighted(img2_kp, 0.7, sm2_color, 0.3, 0)

    match_img = draw_matches(img1_kp, kp1, img2_kp, kp2, matches_inliers)

    cv2.imwrite(os.path.join(args.output_dir, "img1_keypoints.png"), img1_kp)
    cv2.imwrite(os.path.join(args.output_dir, "img2_keypoints.png"), img2_kp)
    cv2.imwrite(os.path.join(args.output_dir, "img1_heatmap.png"), blend1)
    cv2.imwrite(os.path.join(args.output_dir, "img2_heatmap.png"), blend2)
    cv2.imwrite(os.path.join(args.output_dir, "matches.png"), match_img)

    # Summary figure
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"Image 1: {kp1.shape[1]} Keypoints")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
    plt.title(f"Image 2: {kp2.shape[1]} Keypoints")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(blend1, cv2.COLOR_BGR2RGB))
    plt.title("UWNet Score Heatmap (Image 1)")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{matches_inliers.shape[1]} Inlier Matches ({args.ransac_method} RANSAC)")
    plt.axis('off')

    plt.tight_layout()
    summary_path = os.path.join(args.output_dir, "summary.png")
    plt.savefig(summary_path)
    print(f"Results saved to {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
