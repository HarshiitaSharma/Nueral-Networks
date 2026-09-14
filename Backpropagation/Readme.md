# Backpropagation

Objective: Understanding how backpropagation works by implementing a small XOR network from scratch, then compare it with PyTorch's automatic differentiation.

## Learning Objectives
- Implement forward and backward propagation for XOR using NumPy.
- Understand manual weight updates.
- Train the same network using PyTorch.
- Compare important loss functions and optimizers.
- Interpret gradients, weights, and predictions.

## Running it

```bash
pip install torch matplotlib
jupyter notebook Backpropagation.ipynb
```

## Compare Important Loss Functions

| Loss Function | Typical Use | PyTorch | Remarks |
|---|---|---|---|
| **MSELoss** | Regression | `nn.MSELoss()` | Simple squared error |
| **BCELoss** | Binary Classification | `nn.BCELoss()` | Requires sigmoid output |
| **CrossEntropyLoss** | Multi-class Classification | `nn.CrossEntropyLoss()` | Most common multi-class loss |

## Compare Main Optimizers

| Optimizer | PyTorch | Advantage | Use |
|---|---|---|---|
| **SGD** | `torch.optim.SGD` | Simple and memory-efficient | Small models |
| **Adam** | `torch.optim.Adam` | Fast convergence | Default choice for deep learning |
| **RMSprop** | `torch.optim.RMSprop` | Adaptive learning rate | RNNs and noisy gradients |

## Task: Optimizer Comparison

The Adam optimizer was replaced with **SGD** and **RMSprop** to compare the training performance of all three optimizers.

The following steps were performed:

1. Train the model using **Adam**.
2. Train the same model using **SGD**.
3. Train the same model using **RMSprop**.
4. Record the **final loss** after training.
5. Plot **Loss vs. Epoch** for each optimizer.
6. Compare the convergence behavior and final loss of the three optimizers.

### Comparison

| Optimizer | Final Loss | Convergence |
|---|---:|---|
| Adam | — | Fast |
| SGD | — | — |
| RMSprop | — | — |

> **Note:** Replace the `—` values with the final losses obtained during training.

# Challenge Exercise
- Change the learning rate to 0.1 and 1.0, and compare convergence.
- Print the loss every 500 epochs.
- Print the initial and final weights.
- Explain the role of dsigmoid().

# Questions
1. Explain forward propagation.
2. Explain backpropagation from the NumPy code.
3. Why do we use the sigmoid derivative?
4. What does loss.backward() do?
5. What is the difference between the manual implementation and PyTorch?
6. When should MSELoss, BCELoss, and CrossEntropyLoss be used?
7. Compare SGD and Adam.