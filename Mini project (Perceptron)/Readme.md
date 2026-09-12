# Binary Pattern Classifier using Perceptron

## Overview
This project implements a Binary Pattern Classifier using a single-layer Perceptron. The model is trained on the **OR Logic Gate** dataset and deployed using a **Streamlit** web interface.

Users can enter two binary inputs (0 or 1), and the application predicts the output while displaying the raw sigmoid probability, learned weights, and bias.

---

## Features
- Perceptron implementation from scratch using NumPy
- Training on OR logic gate dataset
- Model saved in JSON format
- Streamlit-based graphical interface
- Prediction using binary inputs
- Display of:
  - Predicted class
  - Raw sigmoid probability
  - Learned weights
  - Learned bias

---

## Project Structure

```
BinaryPatternClassifier/
│── train_model.py
│── app.py
│── perceptron_or_gate_model.json
│── README.md
│── Binary Pattern Classifier using Perceptron.docx
```

---

## Requirements

- Python 3.x
- NumPy
- Streamlit

Install the required packages:

```bash
pip install numpy streamlit
```

---

## How to Run

### Step 1: Train the Model

Run:

```bash
python train_model.py
```

This generates the trained model file:

```
perceptron_or_gate_model.json
```

---

### Step 2: Launch the Streamlit Application

Run:

```bash
python -m streamlit run app.py
```

The application will open automatically in your web browser.

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

- Predicted Class
- Raw Sigmoid Probability
- Learned Weights
- Learned Bias

---

## Dataset

The perceptron is trained on the OR Logic Gate.

| Input 1 | Input 2 | Output |
|---------|---------|--------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 1 |

---

## Technologies Used

- Python
- NumPy
- Streamlit
- JSON

---
## Deliverables
- A trained perceptron model saved after running the training loop.
- A Streamlit app with two input widgets (0/1) and a ‘Predict’ button.
- Display of the predicted class and the raw sigmoid probability.
- Display of the learned weights and bias.