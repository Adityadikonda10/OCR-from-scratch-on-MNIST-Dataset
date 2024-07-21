import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from keras.datasets import mnist

with open('OCR_Model_128,64,64,10.pkl', 'rb') as f:
    model_parameters = pickle.load(f)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class ActivationSoftMax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test = X_test / 255.0
X_test = X_test.reshape(X_test.shape[0], -1)

input_layer = Layer_Dense(784, 128, model_parameters["input_layer_weights"], model_parameters["input_layer_biases"])
activation_input = ActivationReLU()

hidden_layer1 = Layer_Dense(128, 64, model_parameters["hidden_layer1_weights"], model_parameters["hidden_layer1_biases"])
activation1 = ActivationReLU()

hidden_layer2 = Layer_Dense(64, 64, model_parameters["hidden_layer2_weights"], model_parameters["hidden_layer2_biases"])
activation2 = ActivationReLU()

output_layer = Layer_Dense(64, 10, model_parameters["output_layer_weights"], model_parameters["output_layer_biases"])
activation_output = ActivationSoftMax()

def forward_pass(X):
    input_layer.forward(X)
    activation_input.forward(input_layer.output)
    
    hidden_layer1.forward(activation_input.output)
    activation1.forward(hidden_layer1.output)
    
    hidden_layer2.forward(activation1.output)
    activation2.forward(hidden_layer2.output)
    
    output_layer.forward(activation2.output)
    activation_output.forward(output_layer.output)
    
    return activation_output.output

def visualize_predictions(X, y_true, y_pred, num_samples=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_true[i]}\nPred: {y_pred[i]}")
        plt.axis('off')
    plt.show()

for _ in range(5):
    indices = np.random.choice(X_test.shape[0], 10, replace=False)
    X_sample = X_test[indices]
    y_sample = Y_test[indices]
    
    predictions = np.argmax(forward_pass(X_sample), axis=1)
    
    visualize_predictions(X_sample, y_sample, predictions, num_samples=10)
