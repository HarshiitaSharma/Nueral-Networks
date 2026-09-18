# Regularization Methods for Neural Networks

## Overview
This project explores **overfitting** in neural networks and compares five techniques used to reduce it, using **PyTorch** and the **FashionMNIST** dataset. A small feedforward network is deliberately trained on a limited dataset so it overfits, and then the same network is retrained with different regularization methods to see how much each one closes the gap between training and test accuracy.

---

## Features
- Baseline network trained with no regularization, to observe overfitting directly
- **L2 regularization** (weight decay) at multiple strengths
- **L1 regularization**, including a check of how many weights get pushed near zero
- **Dropout**, including a from-scratch demonstration of the inverted-dropout formula and what happens if `model.eval()` is skipped
- **DropConnect**, a custom `nn.Module` that randomly zeroes individual weights (rather than activations) during training
- **Batch Normalization**, including a manual reproduction of the BatchNorm calculation
- A combined model (Dropout + BatchNorm + L2) compared against the plain baseline
- Summary table and bar chart comparing the generalization gap across all methods

---

## Project Structure

```
regularization-methods/
│── regularization_methods.ipynb
│── data/                  <- FashionMNIST, downloaded automatically on first run
│── README.md
```

---

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- Pandas
- Seaborn

Install the required packages:

```bash
pip install torch torchvision matplotlib numpy pandas seaborn
```

---

## How to Run

### Step 1: Open the Notebook

```bash
jupyter notebook regularization_methods.ipynb
```

### Step 2: Run All Cells

FashionMNIST is downloaded automatically the first time the data-loading cell runs. Only the first 2,000 training and test images are used, on purpose — a small dataset makes the network overfit quickly, which is the point of the experiment.

### Step 3: Compare Results

Each experiment trains a version of the same base network (784 → 256 → 10) with a different regularization setting and prints its final training/test accuracy and gap. The final cells build a summary table and a bar chart across all runs.

---

## Network Architecture

A single, shared architecture is used throughout so comparisons are fair — only one setting changes at a time:

- **Input:** 784 (28×28 image flattened)
- **Hidden layer:** 256 neurons, ReLU activation
- **Output:** 10 classes (clothing categories)
- Optional: `BatchNorm1d` before the activation, `Dropout` after it

---

## Methods Compared

| Method | What it does |
|---|---|
| Plain (no regularization) | Baseline — shows the overfitting problem |
| L2 (weight decay) | Penalizes large weights via the optimizer |
| L1 | Penalizes weights via an added loss term; pushes many weights to ~0 |
| Dropout | Randomly zeroes activations during training |
| DropConnect | Randomly zeroes individual weights during training |
| Batch Normalization | Normalizes layer inputs during training |
| Combined | Dropout + BatchNorm + L2 together |

---

## Output

For each experiment, the notebook reports:

- Final training accuracy
- Final test accuracy
- Generalization gap (train accuracy − test accuracy)
- Loss curves (training vs. test) for visual comparison

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- Pandas / Seaborn

---

## Key Takeaway

A small network trained on limited data overfits badly (large train/test gap). Each regularization method reduces that gap by a different mechanism — penalizing weight size (L1/L2), adding training-time noise (Dropout/DropConnect), or stabilizing layer inputs (BatchNorm) — and combining several methods gives the best generalization in this experiment.