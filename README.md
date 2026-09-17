# Neural Networks from Scratch

A growing collection of hands-on PyTorch projects exploring how neural networks actually work — starting from a single artificial neuron and building up from there. Each folder is a self-contained mini-project with its own README, notebook(s), and notes.

## Goal

Build intuition for the math and mechanics behind deep learning by implementing things manually first, then comparing against PyTorch's built-in tools. Less "run the tutorial", more "understand why it works."

## Projects

| Folder | Topic | Description |
|---|---|---|
| [`artificial-neuron/`](./BuildingArtificial_Neuron) | Single Artificial Neuron | Tensors, manual neuron math (`z = w·x + b`), activation functions, autograd, and `nn.Linear` comparison |
|[Perceptron, Gradient Descent Backpropagation in PyTorch](./Perceptron,%20Gradient%20Descent%20%26%20Backpropagation/Readme.md) | Perceptron| Gradient Descent| Backpropagation |Logic Gate|
|[Mini project (Perceptron)](./Mini%20project%20(Perceptron)/Readme.md) |Binary Pattern Classifier|Streamlit interface| Logic gates| Predicted class|raw sigmoid probability|
|[Backpropagation](./Backpropagation/Readme.md) |forward propagation|XOR Network| Logic gates| Loss Functions|raw sigmoid probability|MSELoss|BCELoss|CrossEntropyLoss|
| [Backpropagation Project](./Miniproject-Backpropagation/Readme.md) | Backpropagation | Forward propagation and an XOR network trained on logic gates, comparing MSE, BCE, and Cross-Entropy loss functions, with raw sigmoid probability outputs |

## Structure

```
.
├── README.md                  
├── artificial-neuron/
│   ├── README.md
│   └── artificial_neuron_pytorch.ipynb
|
|── Perceptron, Gradient Descent Backpropagation in PyTorch
|   ├── README.md
│   └── Perceptron_gradient_backpropagation.ipynb
|
|── Mini project (Perceptron)
|   ├── README.md
│   ├── perceptron_or_gate_model.ipynb
|   ├── app.py
│   └── train_model.py
|── Backpropagation in PyTorch
|   ├── README.md
│   └── Backpropagation.ipynb
|
|── Mini Project (Backpropagation)
|   ├── README.md
│   └── Backpropagation.ipynb

```

## Setup

Most notebooks only need:

```bash
pip install torch matplotlib
jupyter notebook
```

Individual folders will call out any extra dependencies in their own README.

## Why this repo exists

Learning neural networks by building the small pieces first — a neuron, then a layer, then a network — rather than jumping straight to high-level APIs.
