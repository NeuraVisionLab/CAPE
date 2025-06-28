# CAPE: Connectivity-Aware Path Enforcement Loss for Curvilinear Structure Delineation

This repository contains the implementation of **CAPE (Connectivity-Aware Path Enforcement Loss)**, a novel loss function designed to improve the connectivity of curvilinear structures in biomedical image segmentation, such as neuronal processes and blood vessels. The method leverages Dijkstra's algorithm to enforce topological correctness by comparing shortest paths between ground truth and predicted segmentations, as described in the paper.

 * [Project Page](https://neuravisionlab.github.io/CAPE/)
 * [Paper](https://arxiv.org/abs/2504.00753)

## Overview

CAPE addresses the challenge of preserving topological connectivity in curvilinear structure segmentation, a critical issue in biomedical imaging where conventional pixel-wise loss functions often fail to ensure global connectivity. By computing shortest paths in the pixel domain and comparing their costs, CAPE generates denser gradients along entire paths, enhancing connectivity enforcement while remaining suitable for gradient-based optimization. The implementation supports both 2D and 3D datasets and integrates seamlessly with deep learning frameworks like PyTorch.

## Features

- **Connectivity-Aware Loss**: Optimizes topological correctness using a differentiable loss based on the Average Path Length Similarity (APLS) metric.
- **2D and 3D Support**: Handles both 2D and 3D tasks.
- **Dijkstra-Based Path Computation**: Uses Dijkstra's algorithm to compute shortest paths, with a masking strategy to handle noisy centerlines and loops.
- **Flexible Window-Based Processing**: Processes large images/volumes by dividing them into manageable patches.

## Installation

To use this code, you need to have the following dependencies installed.

```bash
# Clone the repository
git clone https://github.com/neuravisionlab/CAPE.git

# Install dependencies
pip install torch numpy scikit-image opencv-python scipy networkx
```

**Note**: The code requires custom utility functions `graph_from_skeleton_2D` and `graph_from_skeleton_3D`, which are available in the `utils` module. Ensure these are included in your project directory.

## Datasets

The CAPE loss has been evaluated on the following datasets:

- [CREMI](https://cremi.org/data/)
- [DRIVE](https://drive.grand-challenge.org)
- Brain

## Citing

If you find our work useful, please consider citing:

```BibTeX
@misc{esmaeilzadeh2025capeconnectivityawarepathenforcement,
      title={CAPE: Connectivity-Aware Path Enforcement Loss for Curvilinear Structure Delineation}, 
      author={Elyar Esmaeilzadeh and Ehsan Garaaghaji and Farzad Hallaji Azad and Doruk Oner},
      year={2025},
      eprint={2504.00753},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.00753}, 
}
```
