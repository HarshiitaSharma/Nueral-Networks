# Artificial Neuron in PyTorch

A from-scratch walkthrough of how a single artificial neuron works, built and verified in PyTorch — covering tensors, manual neuron math, activation functions, autograd, and a comparison against `nn.Linear`.

## What's inside

- **Tensor basics** — creating and inspecting 1D/2D/3D tensors
- **Tensor arithmetic & dot products**
- **Manual neuron implementation** — computing `z = w · x + b` by hand
- **Activation functions** — Sigmoid, Tanh, ReLU (with plots)
- **Autograd** — computing gradients automatically and checking them against calculus by hand
- **`nn.Linear`** — building the same neuron with PyTorch's built-in layer and comparing outputs to the manual version

## Learning Objectives
• Understand the role of PyTorch in deep learning.
• Create and manipulate tensors.
• Perform tensor operations.
• Implement an artificial neuron from its mathematical equation.
• Apply activation functions using PyTorch.
• Understand automatic differentiation (Autograd).
• Create neurons using nn.Linear.

## Running it

```bash
pip install torch matplotlib
jupyter notebook artificial_neuron_pytorch.ipynb
```

## Why

Wanted a clear, minimal reference for how the math behind a neuron (weighted sum + bias + activation) maps onto actual PyTorch code, before moving on to full networks.

## Tasks
1. Create tensors of different shapes.
2. Perform arithmetic and dot product operations.
3. Implement a neuron with 5 inputs.
4. Change the weights and bias; predict the output before execution.
5. Plot Sigmoid, Tanh and ReLU.

6. Use Autograd to compute the gradient of y = x² + 5x + 2.
7. Create nn.Linear(5, 1) and inspect its weights.
8. Compare the manual neuron implementation with nn.Linear.