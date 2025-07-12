# SignStruct: Structure-Aware Semantic-Guided 3D Hand Reconstruction

---

## Project Overview

**SignStruct** is an advanced, structure-aware framework for reconstructing accurate 3D hand poses and structures from dual-view RGB video streams, specifically designed for egocentric scenarios such as American Sign Language (ASL) gesture analysis.

### Problem Statement

Egocentric 3D hand pose estimation remains challenging due to issues like rapid hand motion, severe occlusion, and motion blur. SignStruct addresses these challenges by integrating stereo vision, semantic guidance, latent structure modeling, and temporal smoothing into a cohesive pipeline.

### Inputs and Outputs

* **Inputs**:

  * Dual-view RGB images (left and right cameras)
  * Camera intrinsic and extrinsic parameters

* **Outputs**:

  * Reconstructed 3D hand joints (21 landmarks per hand)
  * MANO parameters (pose, shape, global transformations)

### Models & Dataset

* **Backbone**: Swin Transformer (Tiny variant)
* **3D Reconstruction**: MANO Hand Model
* **Datasets**: Primarily tested on HOT3D and internally collected ASL datasets

---

## Architecture Overview

SignStruct's pipeline comprises modular components for clarity, scalability, and ease of experimentation:

### Code Structure

* `DualSwinFeatureExtractor.py`: Extracts multi-scale features from left/right views using a Swin Transformer backbone.
* `MultiScaleCrossViewFusion.py`: Performs cross-view fusion through deformable attention and epipolar positional encoding.
* `LatentProcessingModule.py`: Encodes fused features into a structured latent space via a structure-aware VAE and temporal transformer.
* `MANODecoder.py`: Decodes latent representations into MANO parameters and performs forward kinematics to reconstruct 3D joints.
* `mano_layer.py` and `hand_common.py`: Provide utilities and definitions for the MANO hand model and hand landmark management.

### Processing Pipeline

```
Input RGB Images (Dual Views)
            |
            ▼
DualSwinFeatureExtractor (Feature Pyramid Network)
            |
            ▼
MultiScaleCrossViewFusion (Epipolar-Aware Fusion)
            |
            ▼
LatentProcessingModule (Structure-Aware Transformer + VAE)
            |
            ▼
MANODecoder (Latent to MANO Params & 3D Reconstruction)
            |
            ▼
Output 3D Hand Joints & MANO Parameters
```

---

## Results & Metrics

* **Performance Benchmarks**:

  * MPJPE (Mean Per-Joint Position Error)
  * PCK (Percentage of Correct Keypoints @ 10mm)
  * Smoothness and robustness against occlusions

* **Validation**:

  * Run provided unit tests and demo scripts (`MANODecoder.py` has a demo)
  * Verify outputs visually with visualization scripts (optional, provided separately)

---
