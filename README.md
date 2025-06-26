CAPE: Connectivity-Aware Path Enforcement Loss for Curvilinear Structure Delineation
This repository contains the implementation of CAPE (Connectivity-Aware Path Enforcement Loss), a novel loss function designed to improve the connectivity of curvilinear structures in biomedical image segmentation, such as neuronal processes and blood vessels. The method leverages Dijkstra's algorithm to enforce topological correctness by comparing shortest paths between ground truth and predicted segmentations, as described in the paper.

Project Page
Paper

Overview
CAPE addresses the challenge of preserving topological connectivity in curvilinear structure segmentation, a critical issue in biomedical imaging where conventional pixel-wise loss functions often fail to ensure global connectivity. By computing shortest paths in the pixel domain and comparing their costs, CAPE generates denser gradients along entire paths, enhancing connectivity enforcement while remaining suitable for gradient-based optimization. The implementation supports both 2D and 3D datasets and integrates seamlessly with deep learning frameworks like PyTorch.
Features

Connectivity-Aware Loss: Optimizes topological correctness using a differentiable loss based on the Average Path Length Similarity (APLS) metric.
2D and 3D Support: Handles both 2D and 3D biomedical image segmentation tasks.
Dijkstra-Based Path Computation: Uses Dijkstra's algorithm to compute shortest paths, with a masking strategy to handle noisy centerlines and loops.
Integration with U-Net: Designed to work with U-Net architectures for 2D and 3D segmentation tasks.
Flexible Window-Based Processing: Processes large images/volumes by dividing them into manageable patches.

Installation
To use this code, you need to have Python 3.8+ and the following dependencies installed. We recommend setting up a virtual environment.
# Clone the repository
git clone https://github.com/neuravisionlab/CAPE.git
cd CAPE

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy scikit-image opencv-python scipy

Note: The code requires custom utility functions graph_from_skeleton_2D and graph_from_skeleton_3D, which are assumed to be available in the utils module. Ensure these are included in your project directory or implemented as described in the paper.
Usage
The CAPE class is implemented as a PyTorch module and can be integrated into a training pipeline for segmentation tasks. Below is an example of how to use the CAPE loss in a training loop.
import torch
from CAPE import CAPE

# Initialize CAPE loss (2D example)
cape_loss = CAPE(window_size=128, three_dimensional=False, distance_threshold=20, dilation_radius=10)

# Example inputs: predictions and ground truth masks
predictions = torch.randn(1, 128, 128)  # Batch of predicted distance maps
ground_truths = torch.randint(0, 2, (1, 128, 128))  # Batch of binary masks

# Compute CAPE loss
loss = cape_loss(predictions, ground_truths)
print(f"CAPE Loss: {loss.item()}")

# Combine with MSE loss for training
mse_loss = torch.nn.MSELoss()
total_loss = mse_loss(predictions, ground_truths) + 0.1 * loss

For 3D data, set three_dimensional=True and adjust the window_size and dilation_radius as needed (e.g., window_size=48, dilation_radius=5).
Datasets
The CAPE loss has been evaluated on the following datasets:

CREMI: 2D dataset with 83 training and 42 validation samples of size 1250x1250, containing neurons from Drosophila melanogaster brain.
DRIVE: 2D dataset with 13 training and 7 validation samples of size 584x565, containing retinal blood vessels.
Brain: 3D dataset with 14 light microscopy scans of the mouse brain, each of size 250x250x200 (10 for training, 4 for validation).

Evaluation Metrics
The implementation supports evaluation using:

Pixel-wise Metrics: CCQ (Correctness, Completeness, Quality) and Dice score.
Topology-aware Metrics: APLS (Average Path Length Similarity) and TLTS (Too-Long-Too-Short).

See the paper for detailed results comparing CAPE against baselines like MSE, Perceptual loss, clDice, and InvMALIS.
Training Details

Architecture: 2D and 3D U-Net with three down-samplings, two convolutional layers per level, max-pooling (encoder), and bilinear upsampling (decoder).
Optimizer: Adam with a learning rate of 1e-3 and weight decay of 1e-3.
Epochs: 10k for 2D models, 50k for 3D models.
Loss Combination: CAPE loss is combined with MSE loss using a hyperparameter α (e.g., L_TOTAL = L_MSE + α * L_CAPE).

Results
CAPE achieves state-of-the-art performance in topology-aware metrics (APLS, TLTS) while maintaining competitive pixel-wise performance (CCQ, Dice). Qualitative results show significant improvements in connectivity for curvilinear structures.
For detailed quantitative results, refer to Table 1 in the paper.
Citing
If you find our work useful, please consider citing:
@misc{esmaeilzadeh2025capeconnectivityawarepathenforcement,
      title={CAPE: Connectivity-Aware Path Enforcement Loss for Curvilinear Structure Delineation}, 
      author={Elyar Esmaeilzadeh and Ehsan Garaaghaji and Farzad Hallaji Azad and Doruk Oner},
      year={2025},
      eprint={2504.00753},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.00753}, 
}
