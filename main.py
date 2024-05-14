import numpy as np
from perceptron import Perceptron

def main():
    # Example usage
    # Define training data
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 0, 1])

    # Create a perceptron
    perceptron = Perceptron(input_size=2, learning_rate=0.1)

    # Train the perceptron
    for _ in range(100):
        for inputs, label in zip(training_inputs, labels):
            perceptron.train(inputs, label)

    # Test the perceptron
    test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    test_labels = np.array([0, 0, 0, 1])
    acc = perceptron.accuracy(test_inputs, test_labels)
    print("Accuracy:", acc)

if __name__ == "__main__":
    main()