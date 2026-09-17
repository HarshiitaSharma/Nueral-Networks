# Logic Gate Neural Network Predictor

## Overview
This project implements a **Multi-Layer Perceptron (MLP)** using **PyTorch** to learn and predict the output of basic logic gates. Unlike a single-layer perceptron, the MLP includes a hidden layer, which allows it to learn non-linearly separable functions like **XOR**.

The application is menu-driven: users select a logic gate (AND, OR, NAND, NOR, or XOR), train a dedicated model for it, and then enter two binary inputs to get a predicted output along with a confidence score.

---

## Features
- MLP implementation using PyTorch (`nn.Module`), with one hidden layer and Sigmoid activation
- Training supported for **AND, OR, NAND, NOR, and XOR** gates
- Separate model trained and saved per logic gate
- Model saved in `.pth` format after training
- Menu-driven console application
- Prediction using two binary inputs
- Display of:
  - Predicted class (0 or 1)
  - Confidence score (raw sigmoid output)

---

## Project Structure

```
logic-gate-mlp/
│── logic_gate_mlp.ipynb
│── and_gate_model.pth
│── or_gate_model.pth
│── nand_gate_model.pth
│── nor_gate_model.pth
│── xor_gate_model.pth
│── README.md
```

*(model `.pth` files are generated after training and are not included until the notebook is run)*

---

## Requirements

- Python 3.x
- PyTorch
- Matplotlib

Install the required packages:

```bash
pip install torch matplotlib
```

---

## How to Run

### Step 1: Open the Notebook

```bash
jupyter notebook logic_gate_mlp.ipynb
```

### Step 2: Run All Cells

This loads the data, defines the `LogicGateMLP` model, and defines the training/prediction functions.

### Step 3: Launch the Menu

```python
main_menu()
```

Follow the on-screen prompts to:
1. Train a model for a chosen logic gate
2. Predict an output for that gate using two binary inputs
3. Exit

Training a gate generates and saves a model file (e.g. `xor_gate_model.pth`) for reuse.

---

## Input

The application accepts two binary inputs:

- Input 1: 0 or 1
- Input 2: 0 or 1

Example:

| Input 1 | Input 2 |
|---------|---------|
| 0 | 0 |
| 0 | 1 |
| 1 | 0 |
| 1 | 1 |

---

## Output

The application displays:

- Predicted Class (0 or 1)
- Confidence Score (raw sigmoid probability)

---

## Dataset

All gates share the same 2-input truth table; only the target output column changes.

| Input 1 | Input 2 | AND | OR | NAND | NOR | XOR |
|---------|---------|-----|----|------|-----|-----|
| 0 | 0 | 0 | 0 | 1 | 1 | 0 |
| 0 | 1 | 0 | 1 | 1 | 0 | 1 |
| 1 | 0 | 0 | 1 | 1 | 0 | 1 |
| 1 | 1 | 1 | 1 | 0 | 0 | 0 |

**Note:** XOR is not linearly separable, so it can't be learned by a single-layer perceptron — this is the main reason the model uses a hidden layer.

---

## Technologies Used

- Python
- PyTorch
- Matplotlib

---

## Deliverables
- Source code with comments, implementing the MLP and training/prediction logic.
- Trained models saved per logic gate (AND, OR, NAND, NOR, XOR) for reuse.
- Confidence score displayed alongside each prediction.
- Menu-driven interface for training and testing any of the five gates.