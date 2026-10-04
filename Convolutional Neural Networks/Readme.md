# Regularization Methods for Neural Networks
Regularization is a technique used in machine learning to prevent overfitting, which otherwise causes models to perform poorly on unseen data. By adding a penalty for complexity, regularization encourages simpler and more generalizable models.

## Overview
- Set up a PyTorch environment and confirm the N, C, H, W tensor convention.
- Compute a convolution by hand and reproduce it with F.conv2d.
- Verify the output-size formula against real layer shapes for a range of strides and paddings.
- Apply ReLU and read its derivative off autograd.
- Perform max and average pooling and observe how each routes gradient backwards.
- Confirm the hand-derived convolution gradients of Section 9 against autograd.
- Count parameters, receptive fields and multiply-accumulate operations from code.
- Build, train and evaluate a CNN image classifier on CIFAR-10.
- Visualise learned filters and feature maps.

---

## Running it

```bash
pip install torch matplotlib
jupyter notebook cnn.ipynb
```

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
Install the libraries once, from a terminal:
```bash
PYTHON
pip install torch torchvision matplotlib
```

## Lab Exercises
Complete these in the same notebook. Record the output of each and, where the exercise asks for it, a
one-line explanation of why the result came out that way.
1. Change the kernel in Experiment 2 to [[1, 1], [1, 1]] and predict the new feature map before running the cell. What pattern does this filter detect?
2. Add the settings (32, 4, 1, 1) and (7, 3, 2, 0) to Experiment 3. Explain the second result using the floor in H_out = floor((H + 2P - K) / S) + 1.
3. In Experiment 6, change the upstream gradient to [[1, 1], [1, 1]] and describe how the max-pooling gradient changes and how the average-pooling gradient changes.
4. Modify Experiment 7 to use a 3x3 kernel on a 4x4 input with an upstream gradient of all ones.Predict dL/db before running it, then check.
5. Add a third convolution layer (32, 64, 3x3, padding 1) and a third pooling layer to SmallCNN. Work out the new flattened size on paper, adjust fc1, and confirm with a shape trace.
6. Replace both nn.MaxPool2d layers with nn.AvgPool2d, retrain for 5 epochs, and compare the test accuracy. Which pooling rule works better here, and does the difference survive a change of random seed?
7. Set padding=0 in both convolution layers of SmallCNN. Trace the shapes, fix fc1, and state how many pixels of the border were lost.
8. Train for 15 epochs instead of 5 and plot training loss against test accuracy per epoch. Identify the epoch at which the two curves start to diverge.
9. Add transforms.RandomHorizontalFlip() to the training transform only, retrain, and report the change in test accuracy. Explain why this transform is applied to the training set but not the test set.
10. Count the multiply-accumulate operations of conv1 and conv2 using MACs = H_out x W_out x K x K x C_in x C_out, then compare the two figures with the parameter counts printed in Experiment 10. Which layer is the more expensive, and which is the larger?

# Convolutional Neural Network Formulas

| Quantity | Formula | Verified in |
| --- | --- | --- |
| **Convolution (cross-correlation)** | $(I * K)(i, j) = \sum_{m} \sum_{n} I(i+m, j+n) K(m, n)$ | Experiment 2 |
| **ReLU activation** | $f(x) = \max(0, x)$ | Experiment 4 |
| **Output size** | $O = \lfloor \frac{W - K + 2P}{S} \rfloor + 1$ | Experiment 3 |
| **Same padding** | $P = \frac{K - 1}{2} \quad \text{(for odd } K\text{, stride } S=1\text{)}$ | Experiment 3 |
| **Convolution parameters** | $N_{\text{params}} = (K_w \cdot K_h \cdot C_{\text{in}} + 1) \cdot C_{\text{out}}$ | Experiment 8 |
| **Receptive field** | $RF_l = RF_{l-1} + (K_l - 1) \cdot \prod_{i=1}^{l-1} S_i$ | Experiment 8 |
| **Multiply-accumulate count** | $\text{MACs} = H_{\text{out}} \cdot W_{\text{out}} \cdot C_{\text{out}} \cdot (K_w \cdot K_h \cdot C_{\text{in}})$ | Experiment 8 |
| **Bias gradient** | $\frac{\partial L}{\partial b_k} = \sum_{i} \sum_{j} \frac{\partial L}{\partial Y_{i,j,k}}$ | Experiment 7 |
| **Kernel gradient** | $\frac{\partial L}{\partial K} = X * \frac{\partial L}{\partial Y}$ | Experiment 7 |
| **Input gradient** | $\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} * K_{\text{rot180}}$ | Experiment 7 |
| **Max-pooling gradient** | $\frac{\partial L}{\partial X_{i,j}} = \frac{\partial L}{\partial Y_{\text{pool}}} \cdot \mathbb{I}(X_{i,j} = \max(X))$ | Experiment 6 |
| **Average-pooling gradient** | $\frac{\partial L}{\partial X_{i,j}} = \frac{1}{N_{\text{pool}}} \cdot \frac{\partial L}{\partial Y_{\text{pool}}}$ | Experiment 6 |
