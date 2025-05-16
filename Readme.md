# Zero-Shot 6D Pose Estimation and Navigation Pipeline by Srijan Dokania

This repository provides a pipeline for zero-shot 6D pose estimation from monocular RGB images, integrating MC-CLIPSeg segmentation, MiDaS depth estimation, and Weighted PCA for robust pose calculation. The estimated poses can be integrated into a 3D Gaussian Splatting (3DGS) map for robot localization and navigation, as demonstrated in the provided laboratory image. This repository does not have the SplatPlan and SplatNav code and its merging with the proposed zero-shot pipeline

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Input Preparation](#input-preparation)
- [Usage](#usage)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

This pipeline enables:
- Zero-shot segmentation of robots or objects in monocular RGB images using text prompts (CLIPSeg).
- Monocular depth estimation (MiDaS).
- 6D pose estimation via Weighted PCA.
- Integration of pose estimates into a 3D Gaussian Splatting map for robot navigation.
- Batch processing of video frames and output as annotated video.

---

## Requirements

- Python 3.8+
- PyTorch (with CUDA if GPU is available)
- OpenCV (`opencv-python`)
- Transformers (`transformers`)
- NumPy
- tqdm
- torch.hub (for MiDaS)

**Recommended:**  
- NVIDIA GPU with CUDA support for faster inference.

---

## Installation

1. **Clone the repository and navigate to the directory:**
    ```
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2. **Install Python dependencies:**
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or appropriate CUDA version
    pip install opencv-python numpy tqdm transformers
    ```

3. **(Optional, for MiDaS)**
    - The MiDaS model will be automatically downloaded via `torch.hub` on first run.

---

## Input Preparation

1. **Prepare a folder of image frames:**
    - Place your monocular RGB frames in a folder (e.g., `frames/`).
    - Supported formats: `.jpg`, `.png`
    - Naming convention: `frame00001.jpg`, `frame00002.jpg`, ... (or any lexicographically sortable names).

2. **(Optional) Camera Intrinsics:**
    - If using dataset-specific intrinsics (e.g., LineMOD), provide a `scene_camera.json` file.
    - Otherwise, specify camera parameters via command-line arguments.

---

## Usage

### **Single Image Pose Estimation**

```
python zero_shot.py --image frame_000401.jpg --prompt "white ugv robot with wheels"

```
## Example

![Lab Example](highbay_results.png)

*Example of pose estimation and annotation in a highbay environment with three robots.*
