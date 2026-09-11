# Perceptron

A single-layer perceptron is the simplest neural network: it takes a set of inputs, computes a weighted sum plus
a bias, and passes the result through an activation function to produce a binary output. Geometrically, it learns a
straight line (or hyperplane, in higher dimensions) that separates two classes — so it can only solve problems
that are linearly separable.
Training these on Logic gates.

## Learning Objectives
• Explain the perceptron as a single-layer linear binary classifier.
• Implement a perceptron model in PyTorch using nn.Linear and nn.Sigmoid.
• Explain the role of a loss function in measuring prediction error.
• Use PyTorch's autograd engine to compute gradients automatically.
• Explain gradient descent as an iterative weight-update procedure.
• Implement a complete training loop combining forward pass, loss computation, backpropagation and
parameter updates.
• Interpret learned weights, biases and gradients.

## Running it

```bash
pip install torch matplotlib
jupyter notebook Perceptron_gradient_backpropagation.ipynb
```

### Understanding Backpropagation

Backpropagation is how PyTorch's `autograd` engine computes the gradient of the loss with respect to every parameter using the chain rule. In practice, it is triggered by a single call: `loss.backward()`.

| Step | What Happens |
|---|---|
| **1. Forward pass** | Inputs flow through the model to produce a prediction. |
| **2. Compute loss** | `criterion(output, y)` compares the prediction with the true label. |
| **3. `loss.backward()`** | Autograd walks the computation graph backward and computes `∂loss/∂w` for every parameter. |
| **4. Gradients stored** | Each parameter's `.grad` attribute now holds its gradient. |
| **5. `optimizer.step()`** | Every parameter is updated: `w ← w − lr × grad`. |
| **6. `optimizer.zero_grad()`** | Gradients are reset before the next epoch because PyTorch accumulates gradients by default. |


## Challenge Exercises
1. Train the perceptron on the AND gate truth table.
2. Train the perceptron on the OR gate truth table.
3. Attempt to train it on the XOR gate and observe why it fails to converge.
4. Change the learning rate to 0.01 and then to 0.5, and compare convergence speed.
5. Increase the number of epochs to 5000 and observe the effect on the final loss.
6. Replace SGD with torch.optim.Adam and compare convergence behaviour.
7. Print the loss every 100 epochs instead of only at the end.
8. Plot loss versus epoch using matplotlib.
9. Display the learned weights and bias after training.
10. Explain, in your own words, the effect of the Sigmoid activation on the output.

# Question
1. What is a perceptron?
2. Why can't a single perceptron solve the XOR problem?
3. What is a loss function, and why is it needed?
4. What is gradient descent?
5. What does loss.backward() actually do?
6. Why must optimizer.zero_grad() be called every epoch?
7. What is the difference between SGD and Adam?
8. What is an epoch?
9. What is backpropagation, and how does it relate to the chain rule?
10. Why is Sigmoid commonly used in binary classification?