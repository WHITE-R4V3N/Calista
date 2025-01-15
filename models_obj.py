#   Description:    A file that contains all Neural Network model class and/or objects.

import numpy as np
import matplotlib.pyplot as plt

from settings import *

class Models():
    def __init__(self):
        self.models = []

    def add_model(self, model_obj):
        self.models.append(model_obj)

# Model for predicting what tasks should be done
class Predictive_NN():
    def __init__(self, input_size, hidden_size, hidden2_size, output_size, learning_rate=0.001):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))

        self.w2 = np.random.randn(hidden_size, hidden2_size)
        self.b2 = np.zeros((1, hidden2_size))

        self.w3 = np.random.randn(hidden2_size, output_size)
        self.b3 = np.zeros((1, output_size))

        self.learning_rate = learning_rate
        self.training_loss = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.sigmoid(self.z3)

        return self.a3
    
    def backward(self, X, y, output):
        error = y - output
        d_output = error * self.sigmoid_derivative(output)

        error_hidden2 = d_output.dot(self.w3.T)
        d_hidden2 = error_hidden2 * self.sigmoid_derivative(self.a2)

        error_hidden1 = d_hidden2.dot(self.w2.T)
        d_hidden1 = error_hidden1 * self.sigmoid_derivative

        self.w3 += self.a2.T.dot(d_output) * self.learning_rate
        self.b3 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        self.w2 += self.a1.T.dot(d_hidden2) * self.learning_rate
        self.b2 += np.sum(d_hidden2, axis=0, keepdims=True) * self.learning_rate

        self.w1 += X.T.dot(d_hidden1) * self.learning_rate
        self.b1 += np.sum(d_hidden1, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        printProgressBar(0, epochs, prefix='PROGRESS:', suffix='Complete', length=50)

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 100 == 0:
                self.training_loss.append(np.mean((y - output) ** 2))

            printProgressBar(0, epochs, prefix='PROGRESS:', suffix='Complete', length=50)
        
        return self.training_loss

class Transformer:
    def __init__(self, d_model, n_heads, d_ff, seq_len, learning_rate=0.001):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.seq_len = seq_len
        self.learning_rate = learning_rate

        self.w_q = [np.random.randn(d_model // n_heads, d_model // n_heads) for _ in range(n_heads)]
        self.w_k = [np.random.randn(d_model // n_heads, d_model // n_heads) for _ in range(n_heads)]
        self.w_v = [np.random.randn(d_model // n_heads, d_model // n_heads) for _ in range(n_heads)]

        self.w_o = np.random.randn(d_model, d_model)

        self.w1 = np.random.randn(d_model, d_ff)
        self.w2 = np.random.randn(d_ff, d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        scores = np.dot(Q, K.T) / np.sqrt(Q.shape[-1])
        weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

        return np.dot(weights, V)

    def multi_head_attention(self, X):
        pass

# Runs the AI model 1000 times to test network loss and accuracy
def nn_thousand_test(model_obj):
    counter = 0
    accuracy_array = []

    while counter < 1000:
        # Add code for calling the AI here

        # accuracy_array.append(final_loss)
        counter += counter

# Used to visualize the neural network
def visualize_models(model_obj):
    pass
