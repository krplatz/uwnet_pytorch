# uwnet\_pytorch

## Overview

**This repository is a PyTorch implementation replicating only the UWNet feature detection model from the RU‑SLAM system (Wang et al., Sensors 2024).**

A PyTorch-based framework for generating, training, and evaluating underwater image matching models. It includes:

- **Data loading** for SUNRGBD and pseudo-underwater images
- **Pseudo-underwater generation** using an improved Jaffe–McGlamery model
- **Teacher–Student distillation**: SuperPoint teacher → UWNet student
- **UWNet architecture**: local keypoint detection + global descriptors (NetVLAD)
- **Inference** and **tracking** scripts for visualization and matching
- **Evaluation** on HPatches benchmark

## Repository Structure

```
imwm_project/
├── data/
│   ├── paired_dataloader.py   # Paired SUNRGBD + pseudo-underwater loader
│   ├── pseudo_dataloader.py   # Pseudo-underwater loader
│   └── sun_dataloader.py      # SUNRGBD nested loader
│
├── models/
│   ├── netvlad.py             # NetVLAD layer implementations
│   ├── superpoint_pytorch.py  # Dense SuperPoint implementation
│   ├── teacher_model.py       # SuperPoint-based teacher
│   └── uwnet.py               # UWNet student architecture
│
├── utils/
│   ├── losses.py              # Distillation loss functions
│   └── pseudo_underwater.py   # Pseudo-underwater generator
│
├── camera_test.py            # Webcam capture & display demo
├── evaluate_hpatches.py      # HPatches evaluation script
├── generate_pseudo.py        # Pseudo-underwater image generator
├── inference.py              # UWNet inference & keypoint visualization
├── train_preprocessed.py     # Distillation training pipeline
└── uwnet_tracker.py          # Keypoint tracking & matching demo
```

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy
- Pillow
- matplotlib
- tqdm
- scikit-learn (for NetVLAD init, optional)

Install with:

```bash
pip install torch torchvision opencv-python numpy pillow matplotlib tqdm scikit-learn
```

## Usage

### 1. Generate Pseudo-Underwater Images

```bash
python generate_pseudo.py --output_dir pseudo_images --max_samples 1000
```

Generates pseudo-underwater images from SUNRGBD samples.

### 2. Train UWNet via Distillation

```bash
python train_preprocessed.py \
  --sun_root /path/to/SUNRGBD \
  --pseudo_dir pseudo_images \
  --epochs 150 \
  --batch_size 16
```

Trains UWNet student under guidance of SuperPoint teacher.

### 3. Inference & Visualization

```bash
python inference.py --model_path UWNet_v7.pth --image_dir pseudo_images --threshold 0.0015
```

Runs UWNet on images / webcam frames and shows keypoints + heatmaps.

### 4. Keypoint Tracking & Matching

```bash
python uwnet_tracker.py \
  --model_path models/UWNet_v9.pth \
  --image1 pseudo_images/NYU0001.png \
  --image2 pseudo_images/NYU0002.png \
  --ransac_method fundamental
```

Detects and matches keypoints between two frames with RANSAC filtering.

### 5. HPatches Evaluation

```bash
python evaluate_hpatches.py --method uwnet --dataset /path/to/HPatches --output_dir results
```

Evaluates repeatability, matching score, and MMA on HPatches sequences.

### 6. Camera Test

```bash
python camera_test.py
```

Opens two windows: live color and grayscale webcam feed.

## Modules Description

- **data/**: three loaders for SUNRGBD, pseudo images, and paired datasets
- **models/netvlad.py**: two variants of NetVLAD layers (simple & sklearn-based)
- **models/superpoint\_pytorch.py**: dense-score and descriptor head
- **models/teacher\_model.py**: loads SuperPoint weights, outputs dense maps + global descriptor
- **models/uwnet.py**: MobileNetV2 backbone split, local DCN+attention, global NetVLAD
- **utils/pseudo\_underwater.py**: improved underwater appearance simulation
- **utils/losses.py**: adaptive distillation loss (Eq. 8)

## Contributing

Feel free to open issues, submit PRs, or suggest improvements—especially on:

- Underwater appearance realism
- Alternative backbones or descriptor heads
- More efficient matching/tracking
- HPatches metrics and plotting enhancements

---

*Empowering reliable underwater image matching through teacher–student learning.*

