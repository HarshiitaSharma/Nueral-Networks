# Training a CNN: Hyperparameter Optimization with PyTorch

## Overview
This project explores **hyperparameter optimization (HPO)** for a convolutional neural network using **PyTorch** and the **CIFAR-10** dataset. Rather than treating tuning as guesswork, each experiment isolates one idea (learning rate, schedules, batch size, search strategy) and checks the underlying formula numerically before applying everything together in a real search.

The lab separates two kinds of quantity: **trainable parameters** (weights and biases, learned by gradient descent) and **hyperparameters** (learning rate, batch size, weight decay, dropout, epochs), which must be fixed before training and chosen by search because the outer objective has no usable gradient.

---

## Features
- A small CNN (3 conv blocks + 2 fully connected layers) used to count parameters vs. hyperparameters
- **Learning-rate stability window** on a 1D quadratic loss, with analytical vs. simulated convergence
- **Learning-rate schedules** (step decay, exponential decay, cosine annealing, linear warmup) written by hand and verified against PyTorch's `StepLR` and `CosineAnnealingLR`
- **Batch size** analysis: updates per epoch, the linear learning-rate scaling rule, and an empirical check that gradient variance falls as 1/B
- **Grid search vs. random search**, including the confidence formula for random search and a Monte Carlo check
- **Expected Improvement** (Bayesian optimization acquisition function) computed in closed form
- **Successive Halving** budget planning, then a real search on CIFAR-10 over learning rate, weight decay and dropout
- Four lab exercises completed at the end of the notebook

---

## Project Structure

```
cnn-hyperparameter-optimization/
│── MCO23378_LAB10.ipynb
│── data/                  <- CIFAR-10, downloaded automatically on first run
│── README.md
```

---

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib

Install the required packages:

```bash
pip install torch torchvision matplotlib
```

A GPU is optional. The notebook uses CUDA if available and falls back to CPU otherwise.

---

## How to Run

### Step 1: Open the Notebook

```bash
jupyter notebook MCO23378_LAB10.ipynb
```

### Step 2: Run All Cells in Order

Run the experiments top to bottom, since helper functions defined early (e.g. `gradient_descent`, `rungs`, `trials_for_confidence`, `SmallCNN`) are reused later, including in the exercises.

CIFAR-10 (about 170 MB) is downloaded automatically the first time the data cell in Experiment 8 runs, so that cell can take a while on a slow connection.

### Step 3: Review Results

Every experiment prints its numbers so they can be compared with the formulas in the lab manual. Experiment 8 prints the full successive-halving log and the winning configuration.

---

## Network Architecture

`SmallCNN` is the single model used throughout:

| Stage | Layer | Output shape |
|---|---|---|
| Input | RGB image | 3 × 32 × 32 |
| Block 1 | Conv(3→32, 3×3) → ReLU → MaxPool | 32 × 16 × 16 |
| Block 2 | Conv(32→64, 3×3) → ReLU → MaxPool | 64 × 8 × 8 |
| Block 3 | Conv(64→128, 3×3) → ReLU → MaxPool | 128 × 4 × 4 |
| Flatten | | 2048 |
| FC 1 | Linear(2048→256) → ReLU → Dropout | 256 |
| FC 2 | Linear(256→10) | 10 classes |

The model has **10 trainable tensors** and **620,362 trainable scalars**.

---

## Experiments

| # | Topic | What it demonstrates |
|---|---|---|
| 1 | Parameters and hyperparameters | 620,362 parameters vs. 6 hyperparameters (`learning_rate`, `batch_size`, `optimizer`, `weight_decay`, `dropout`, `epochs`) |
| 2 | Learning-rate stability window | Gradient descent on J(w) = ½aw² converges only if 0 < η < 2/a; fastest at η = 1/a |
| 3 | Learning-rate schedules | Hand-written schedules match PyTorch's built-in schedulers exactly |
| 4 | Batch size | Updates per epoch = ⌈N/B⌉; gradient variance drops roughly as 1/B |
| 5 | Grid vs. random search | Grid cost multiplies across dimensions; random search needs n ≥ ln(1−P) / ln(1−p) trials |
| 6 | Expected Improvement | Balances a low predicted mean (exploitation) against high uncertainty (exploration) |
| 7 | Successive Halving | Every rung costs about the same; large savings over brute force |
| 8 | Real search on CIFAR-10 | Random sampling of 9 configurations plus Successive Halving (η = 3) |

### Experiment 8 setup
- **Data:** 5,000 CIFAR-10 images for training and 2,000 for validation, both carved from the training split. The test split is deliberately never loaded, so it stays untouched for a single final evaluation.
- **Search space:** learning rate sampled log-uniformly in [10⁻⁴, 10⁻¹·⁵], weight decay log-uniformly in [10⁻⁶, 10⁻³], dropout linearly in [0, 0.6]
- **Search procedure:** 9 random configurations, 3 rungs (9 configs × 1 epoch → 3 configs × 3 epochs → 1 config × 9 epochs)

---

## Key Results

| Experiment | Result |
|---|---|
| Stability window (a = 4) | Stable for 0 < η < 0.5; fastest at η = 0.25 |
| Iterations to reach \|w\| ≤ 0.01 | Formula and simulation agree: 56 at η = 0.02, 3 at η = 0.2 |
| Gradient variance (B = 8 → 128) | Measured drop of about 15.5× vs. 16× predicted by 1/B |
| Random search, p = 0.10 | 20 trials give 87.8% chance of a hit; 29 trials for 95% confidence |
| Expected Improvement | Candidate A (EI ≈ 0.0705) edges out candidate B (EI ≈ 0.0650) |
| Successive Halving, n = 27 | 108 epoch-units vs. 729 brute force (6.75× saving) |
| Successive Halving, n = 81 | 405 epoch-units vs. 6,561 brute force (16.2× saving) |

**Best configuration found on CIFAR-10:** learning rate ≈ 1.51×10⁻³, weight decay ≈ 2.00×10⁻⁶, dropout ≈ 0.26, with a validation loss of **1.2472**. The search cost **27 epoch-units**, compared with 81 for training all nine configurations for 9 epochs each.

---

## Lab Exercises

1. **Stability boundary:** at η = 0.50 the iterates bounce between +1 and −1 forever (|1 − ηa| = 1), while at η = 0.49 they still oscillate but shrink toward 0 (|1 − ηa| = 0.96).
2. **Steeper curvature:** with a = 100 the stability range tightens to 0 < η < 0.02 and the fastest rate becomes 0.01, so one very large curvature forces a small learning rate for the whole model.
3. **Random vs. grid trials:** reaching 99% confidence with a 5% good region needs a fixed number of random trials, far fewer than the 5⁴ = 625 trials of a 4-hyperparameter grid with 5 values each.
4. **Reduction factors:** Successive Halving with n = 64 is compared for η = 2 and η = 4. A larger reduction factor is cheaper but discards more configurations early, which risks dropping a slow starter that would have won later.

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib

---

## Key Takeaway

Hyperparameters cannot be learned by gradient descent, so they must be searched. The learning rate has a hard stability limit set by curvature (η < 2/a), schedules and batch size trade speed against noise, and random search beats grid search as dimensions grow. Successive Halving then makes the search cheap by giving many configurations a small budget and only training the survivors fully. In this lab it found a good CIFAR-10 configuration for a third of the cost of training everything to the same depth.