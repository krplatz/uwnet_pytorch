import cv2
import numpy as np
import random

# Dictionary of Jerlov water types (simplified)
JERLOV_WATER_TYPES = {
    'type_I':  {'r': 0.058, 'g': 0.032, 'b': 0.011},
    'type_II': {'r': 0.102, 'g': 0.044, 'b': 0.015},
    'type_III':{'r': 0.187, 'g': 0.078, 'b': 0.025},
    'type_1C': {'r': 0.289, 'g': 0.108, 'b': 0.034},
    'type_3C': {'r': 0.334, 'g': 0.169, 'b': 0.068}
}

def gaussian_psf_blur(img, kernel_size=5, sigma=1.0):
    """Approximate the point spread function with a Gaussian blur."""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def morphological_close_depth(depth_map, kernel_size=3):
    """
    Morphological close on the depth map to fill small holes.
    Expects depth_map in float [0, ~10], 
    so we scale to 8-bit for morphological ops, then scale back.
    """
    scale_factor = 25.5  # or 25.0, just to get a good 8-bit range for ~10m
    depth_8u = (depth_map * scale_factor).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(depth_8u, cv2.MORPH_CLOSE, kernel)
    return closed.astype(np.float32) / scale_factor

def apply_clahe_to_rgb(img_bgr):
    """
    Apply CLAHE (local contrast enhancement) to an RGB image.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def pseudo_underwater_generator(
    air_img,
    depth_map,
    water_type='type_I',
    eta=None,
    L=1.0,
    psf_kernel_size=11,
    psf_sigma=2.0,
    noise_std=0.02,
    randomize=False,
    use_clahe=True
):
    """
    Generate a pseudo-underwater image using an improved Jaffe–McGlamery model 
    with channel-wise attenuation, color cast, PSF blur, noise, morphological 
    depth smoothing, narrower random ranges, better brightness correction, 
    and optional CLAHE.

    Args:
        air_img (np.ndarray): RGB air image in uint8, shape (H, W, 3).
        depth_map (np.ndarray): Depth map in meters, shape (H, W).
        water_type (str): Which Jerlov water type to simulate if randomize=False.
        eta (tuple): Ambient light color (r, g, b) in [0,1]. If None, picks a bluish default with less red.
        L (float): Illumination intensity, typically 1.0.
        psf_kernel_size (int): Kernel size for PSF Gaussian blur.
        psf_sigma (float): Sigma for PSF Gaussian blur.
        noise_std (float): Std dev of Gaussian noise.
        randomize (bool): If True, override some parameters randomly for each call.
        use_clahe (bool): If True, apply CLAHE after brightness correction for local contrast.

    Returns:
        np.ndarray: Pseudo-underwater image in uint8, shape (H, W, 3).
    """

    # ---------------------------------------------------------------------
    # 1) Randomization logic
    # ---------------------------------------------------------------------
    if randomize:
        # Randomly pick a water type
        water_type = random.choice(list(JERLOV_WATER_TYPES.keys()))
        
        # Random color cast in a narrower range (reduce red channel further)
        if eta is None:
            eta_r = random.uniform(0.45, 0.55)
            eta_g = random.uniform(0.65, 0.80)
            eta_b = random.uniform(0.85, 0.95)
            eta = (eta_r, eta_g, eta_b)

        # Random PSF kernel size & sigma
        psf_kernel_size = random.choice([5, 7, 9, 11])
        psf_sigma = random.uniform(1.0, 2.0)

        # Random noise
        noise_std = random.uniform(0.01, 0.02)

        # Random depth scaling (narrower range)
        depth_scale = random.uniform(0.9, 1.2)
        depth_map = depth_map * depth_scale

    # ---------------------------------------------------------------------
    # 2) Depth clamp & morphological close
    # ---------------------------------------------------------------------
    depth_map = np.clip(depth_map, 0.0, 10.0)  # clamp to [0,10] meters
    # Replace zero-depth or near-zero with small positive
    depth_map[depth_map < 0.01] = 0.01
    # Morphological close to fill small holes
    depth_map = morphological_close_depth(depth_map, kernel_size=3)

    # ---------------------------------------------------------------------
    # 3) Convert air image to float [0,1]
    # ---------------------------------------------------------------------
    air_img_float = air_img.astype(np.float32) / 255.0
    H, W, _ = air_img.shape

    # ---------------------------------------------------------------------
    # 4) Check water type & build beta dict
    # ---------------------------------------------------------------------
    if water_type not in JERLOV_WATER_TYPES:
        raise ValueError(f"Unknown water type: {water_type}")
    beta_dict = JERLOV_WATER_TYPES[water_type]
    
    # ---------------------------------------------------------------------
    # 5) If eta is still None, pick default bluish with reduced red
    # ---------------------------------------------------------------------
    if eta is None:
        eta = (0.5, 0.8, 1.0)  # lower red than before

    # ---------------------------------------------------------------------
    # 6) Apply the improved model channel by channel
    # ---------------------------------------------------------------------
    result = np.zeros_like(air_img_float)

    for c, channel in enumerate(['r','g','b']):
        beta_val = beta_dict[channel]
        # transmittance t = exp(-beta * d)
        t = np.exp(-beta_val * depth_map)

        # direct term
        direct = L * eta[c] * air_img_float[..., c] * t
        # background scattering
        background = L * eta[c] * (1 - t)

        # blur the direct term
        direct_blurred = gaussian_psf_blur(direct, kernel_size=psf_kernel_size, sigma=psf_sigma)

        result[..., c] = direct_blurred + background

    # ---------------------------------------------------------------------
    # 7) Add Gaussian noise
    # ---------------------------------------------------------------------
    noise = np.random.normal(0, noise_std, (H, W, 3)).astype(np.float32)
    result += noise

    # ---------------------------------------------------------------------
    # 8) Clip to [0,1]
    # ---------------------------------------------------------------------
    result = np.clip(result, 0.0, 1.0)

    # ---------------------------------------------------------------------
    # 9) Mild brightness correction (tighter thresholds)
    # ---------------------------------------------------------------------
    mean_brightness = np.mean(result)
    if mean_brightness > 0.7:
        factor = 0.7 / mean_brightness
        result *= factor
    elif mean_brightness < 0.3:
        factor = 0.3 / mean_brightness
        result *= factor

    result = np.clip(result, 0.0, 1.0)

    # ---------------------------------------------------------------------
    # 10) Optional local contrast enhancement (CLAHE)
    # ---------------------------------------------------------------------
    final_u8 = (result * 255).astype(np.uint8)
    if use_clahe:
        final_u8 = apply_clahe_to_rgb(final_u8)

    return final_u8
