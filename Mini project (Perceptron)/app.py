import streamlit as st
import numpy as np
import json

# -----------------------------
# Perceptron Class
# -----------------------------
class Perceptron:

    def __init__(self, num_inputs, learning_rate=0.1, epochs=1000):

        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):

        linear_output = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(linear_output)

    @staticmethod
    def load_model(filename):

        with open(filename, "r") as f:
            model = json.load(f)

        p = Perceptron(
            len(model["weights"]),
            model["learning_rate"],
            model["epochs"]
        )

        p.weights = np.array(model["weights"])
        p.bias = np.array([model["bias"]])

        return p


# -----------------------------
# Load Model
# -----------------------------
perceptron = Perceptron.load_model("perceptron_or_gate_model.json")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Binary Pattern Classifier")

st.write("### OR Gate Perceptron")

input1 = st.selectbox("Input 1", [0,1])

input2 = st.selectbox("Input 2", [0,1])

if st.button("Predict"):

    x = np.array([input1,input2])

    probability = perceptron.predict(x)

    prediction = 1 if probability >= 0.5 else 0

    st.success(f"Predicted Class : {prediction}")

    st.write("### Raw Sigmoid Probability")
    st.write(f"{float(probability):.4f}")

    st.write("### Learned Weights")
    st.write(perceptron.weights)

    st.write("### Learned Bias")
    st.write(float(perceptron.bias))