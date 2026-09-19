# Autoencoders in PyTorch

## Overview
This project implements and compares three types of **autoencoders** using **PyTorch**, trained on the **MNIST** handwritten digit dataset: a standard autoencoder, a denoising autoencoder, and a variational autoencoder (VAE). It explores how these networks learn compact latent representations of images and how that latent space can be used to reconstruct, denoise, and generate new digit images.

---

## Features
- **Standard Autoencoder** — encoder/decoder network that compresses 28×28 images into a small latent vector and reconstructs them
- **Denoising Autoencoder** — trained to reconstruct clean images from artificially corrupted (noisy) inputs
- **Bottleneck size experiment** — trains multiple autoencoders with different latent dimensions (2, 8, 32, 128) and compares reconstruction quality
- **Variational Autoencoder (VAE)** — learns a probabilistic latent space (mean + variance) instead of a fixed vector, combining reconstruction loss with KL divergence
- Generation of new, digit-like images by sampling from the VAE's latent space
- Latent space interpolation between two digits
- Visualization of latent space clustering by digit class

---

## Project Structure

```
autoencoders/
│── autoencoders_pytorch.ipynb
│── data/                  <- MNIST, downloaded automatically on first run
│── README.md
```

---

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- NumPy

Install the required packages:

```bash
pip install torch torchvision matplotlib numpy
```

---

## How to Run

### Step 1: Open the Notebook

```bash
jupyter notebook autoencoders_pytorch.ipynb
```

### Step 2: Run All Cells

MNIST is downloaded automatically the first time the data-loading cell runs. Cells run in order:
1. Train the standard autoencoder and view reconstructions
2. Train the denoising autoencoder and compare noisy vs. reconstructed vs. clean images
3. Run the bottleneck-size experiment across several latent dimensions
4. Train the VAE, generate new digits, and visualize the latent space

---

## Architectures

**Standard / Denoising Autoencoder**
- Encoder: `784 → 128 → latent_dim` (ReLU)
- Decoder: `latent_dim → 128 → 784` (ReLU, Sigmoid output)
- Loss: MSE between input and reconstruction

**Variational Autoencoder (VAE)**
- Encoder: `784 → 400` (ReLU), then separate `fc_mu` and `fc_logvar` heads
- Reparameterization trick to sample from the learned distribution
- Decoder: `latent_dim → 400 → 784` (ReLU, Sigmoid output)
- Loss: Binary Cross-Entropy (reconstruction) + KL Divergence (regularizes latent space toward a standard normal)

---

## Output

The notebook produces:

- Training loss curves for each model
- Side-by-side comparisons of original vs. reconstructed images
- Noisy vs. denoised vs. clean image comparisons
- Reconstruction quality across different latent dimensions
- Newly generated digit images sampled from the VAE
- A 2D scatter plot of the latent space, colored by digit class

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- Matplotlib
- NumPy

---
## Tasks
1. Run the complete undercomplete autoencoder using latent_dim = 32.
2. Submit the training-loss graph.
3. Submit a figure showing at least 8 original and reconstructed images.
4. Repeat the experiment with latent_dim = 2 and 128.
5. Create a small table containing latent dimension, final reconstruction loss, and your visual observation.
6. Answer the viva questions in Section 10.
7. Optional: implement the denoising extension and show noisy input, reconstructed output, and clean target.

## Viva / Conceptual Questions
1. Why can an autoencoder be trained without class labels?
2. What is the purpose of the bottleneck layer?
3. Why is this model called an undercomplete autoencoder?
4. What would happen if the latent dimension were 784 or larger without any regularization?
5. Why is the output layer using Sigmoid in this experiment?
6. Why is MSE suitable for the reconstruction loss used here?
7. During backpropagation, which parameters are updated?
8. Why do we compare the reconstructed image with the original image rather than with the digit label?
9. What happens to reconstruction quality when the latent dimension is made very small?
10. How is a denoising autoencoder different from a standard autoencoder?
11. How can reconstruction error be used for anomaly detection?
12. How is an autoencoder different from PCA?
13. Why can a nonlinear autoencoder represent structures that PCA cannot?
## Key Takeaway

A larger latent dimension (bottleneck) gives an autoencoder more room to preserve detail, so reconstruction loss drops as latent size grows — but at the cost of a less compressed representation. The VAE trades some reconstruction sharpness for a smooth, structured latent space, which is what makes it possible to sample entirely new, coherent digit images rather than only reconstructing ones it has seen.