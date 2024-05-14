import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate):
        self.weights = np.zeros(input_size + 1)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, inputs, label):
        prediction = self.predict(inputs)
        error = (label - prediction)
        self.weights[1:] += self.learning_rate * error * inputs
        self.weights[0] += self.learning_rate * error

    def accuracy(self, test_inputs, test_labels):
        predictions = self.predict(test_inputs)
        accuracy = np.sum(predictions == test_labels) / len(test_inputs)
        return accuracy