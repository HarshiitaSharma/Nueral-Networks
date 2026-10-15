# Training and Understanding CNNs: Weight Initialization and Visualization

## Overview
This project looks at two questions about convolutional neural networks using **PyTorch** and the **CIFAR-10** dataset: *how should the weights start?* and *what has the trained network actually learned?*

In the first half, LeCun, Xavier and He initialization are written from scratch, checked against `torch.nn.init`, and compared on a deep CNN with no BatchNorm, so that the initializer alone decides whether signals survive the forward pass. In the second half, the trained network is opened up with filter plots, feature maps and five attribution methods from **Captum**.

---

## Features
- Fan-in / fan-out calculation for `Conv2d` and `Linear` weights, checked against PyTorch
- **LeCun, Xavier and He** initialization from scratch, in both normal and uniform variants
- Comparison of the scratch versions with `nn.init.kaiming_*` and `nn.init.xavier_*`
- Activation-scale tracking through a 6-layer CNN under five initialization schemes
- Training comparison showing how initialization affects early learning
- **First-layer filter** visualization before and after training
- **Feature maps** captured with forward hooks at an early and a late layer
- **Saliency, Integrated Gradients and Guided Backprop** attribution maps
- **Grad-CAM and Guided Grad-CAM** for class-specific localization
- **Occlusion** maps, plus a timing comparison of all attribution methods
- Three lab exercises on Grad-CAM targets, weight randomization and occlusion settings

---

## Project Structure

```
cnn-initialization-visualization/
│── Mco23378_LAB11.ipynb
│── data/                  <- CIFAR-10, downloaded automatically on first run
│── README.md
```

---

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- Captum

Install the required packages:

```bash
pip install torch torchvision matplotlib numpy captum
```

A GPU is optional. The notebook uses CUDA if available and falls back to CPU otherwise.

---

## How to Run

### Step 1: Open the Notebook

```bash
jupyter notebook Mco23378_LAB11.ipynb
```

### Step 2: Run All Cells in Order

Later cells reuse earlier definitions (`fans`, `he_init`, `apply_init`, `CNN`, `train`, `show`, `to_map`), so run the notebook top to bottom. CIFAR-10 (about 170 MB) is downloaded on the first run of the data-loading cell, which can take a while on a slow connection.

### Step 3: Review Results

Experiment 1 prints tables of weight standard deviations, activation scales and test accuracies. Experiment 2 produces the filter, feature-map and attribution figures, followed by the three exercises.

---

## Network Architecture

The `CNN` class has **six 3×3 conv layers with ReLU and no BatchNorm**, with a max-pool after every second conv, followed by one linear layer.

| Stage | Layers | Output shape |
|---|---|---|
| Input | RGB image | 3 × 32 × 32 |
| Block 1 | Conv(3→32) → ReLU → Conv(32→32) → ReLU → MaxPool | 32 × 16 × 16 |
| Block 2 | Conv(32→64) → ReLU → Conv(64→64) → ReLU → MaxPool | 64 × 8 × 8 |
| Block 3 | Conv(64→128) → ReLU → Conv(128→128) → ReLU → MaxPool | 128 × 4 × 4 |
| Classifier | Flatten → Linear(2048→10) | 10 classes |

BatchNorm is left out on purpose, so the scale of the activations depends only on how the weights were initialized.

---

## Experiments

### Experiment 1: Weight Initialization

| Method | Weight variance | Suited to |
|---|---|---|
| LeCun | 1 / fan_in | Linear or tanh-like activations |
| Xavier (Glorot) | 2 / (fan_in + fan_out) | Tanh / sigmoid |
| He (Kaiming) | 2 / fan_in | ReLU |

Sections 1.1 to 1.5 cover fan computation, the three schemes from scratch, the built-in equivalents, the effect on activations through the network, and the effect on training.

### Experiment 2: Visualizing a Trained CNN

| Section | Technique | What it shows |
|---|---|---|
| 2.1 | First-layer filters | Random noise before training, edge and colour detectors after |
| 2.2 | Feature maps (forward hooks) | What an early layer (32×32) and a late layer (8×8) respond to |
| 2.3 | Saliency, Integrated Gradients, Guided Backprop | Which input pixels affect the class score |
| 2.4 | Grad-CAM, Guided Grad-CAM | Coarse class-specific regions, sharpened with pixel detail |
| 2.5 | Occlusion + timings | How hiding image patches changes the class score, and the cost of each method |

The network for Experiment 2 is He-initialized and trained for 3 epochs on 10,000 images. The attribution examples use the first correctly classified test image, a **cat**.

---

## Key Results

**Activation scale (RMS after each ReLU, layers 1 to 6):**

| Scheme | relu1 | relu6 | Behaviour |
|---|---|---|---|
| PyTorch default | 3.98e-01 | 7.86e-03 | Shrinks steadily |
| LeCun | 6.88e-01 | 1.60e-01 | Shrinks, mildly |
| Xavier | 2.85e-01 | 4.41e-02 | Shrinks |
| He | 9.73e-01 | 1.28e+00 | Stays near 1 |
| std = 0.01 | 3.58e-02 | 4.67e-06 | Vanishes |

**Test accuracy after 2 epochs (5,000 training images, 2,000 test images):**

| Scheme | Accuracy |
|---|---|
| PyTorch default | 9.90% |
| LeCun | 28.30% |
| Xavier | 25.50% |
| He | 37.05% |
| std = 0.01 | 9.90% |

- The two schemes whose signal vanished (PyTorch default and std = 0.01) stay at chance level (about 10%) in this short run.
- The scratch implementations match PyTorch's built-ins to within sampling noise (for example, He standard deviation 0.05900 vs. 0.05891).
- The Experiment 2 network reaches **51.6%** test accuracy after 3 epochs.

**Cost of each attribution method (one image, CPU):**

| Method | Time |
|---|---|
| Grad-CAM | 4.2 ms |
| Guided Backprop | 8.3 ms |
| Saliency | 10.3 ms |
| Integrated Gradients | 212.7 ms |
| Occlusion | 890.9 ms |

Gradient-based methods need one backward pass. Integrated Gradients averages many, and Occlusion needs a full forward pass per patch position (14 × 14 = 196 for a 5×5 window at stride 2).

---

## Lab Exercises

1. **Target class sensitivity in Grad-CAM:** Grad-CAM is run for the true class (cat) and for two other classes (airplane and truck) to see whether the highlighted region shifts with the target.
2. **Guided Backprop weight randomization:** Guided Backprop is run on the trained model and on a randomly initialized copy, as a sanity check of whether the attribution really depends on the learned weights.
3. **Occlusion window and stride:** A 3×3 window at stride 1 (30 × 30 = 900 forward passes) is compared with an 8×8 window at stride 4 (7 × 7 = 49 passes), showing the trade-off between map resolution and compute time.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Captum
- Matplotlib

---

## Key Takeaway

Initialization decides whether a deep network without normalization can learn at all: He initialization keeps ReLU activations at a steady scale, while poor choices let the signal shrink toward zero layer by layer. Once trained, a CNN can be inspected from several angles. Filters and feature maps show what it detects, and attribution methods show which pixels drive a prediction. They differ widely in cost, from a few milliseconds for Grad-CAM to nearly a second for Occlusion on a single image.