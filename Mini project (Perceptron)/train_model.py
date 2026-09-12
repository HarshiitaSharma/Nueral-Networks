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

    def train(self, training_inputs, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):

                prediction = self.predict(inputs)
                error = label - prediction

                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

    def save_model(self, filename):

        model = {
            "weights": self.weights.tolist(),
            "bias": self.bias.item(),
            "learning_rate": self.learning_rate,
            "epochs": self.epochs
        }

        with open(filename, "w") as f:
            json.dump(model, f)


# -----------------------------
# OR Gate Dataset
# -----------------------------
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([0,1,1,1])

# -----------------------------
# Train Model
# -----------------------------
perceptron = Perceptron(2)

perceptron.train(X, y)

perceptron.save_model("perceptron_or_gate_model.json")

print("Training Complete!")
print("Weights :", perceptron.weights)
print("Bias :", perceptron.bias.item())