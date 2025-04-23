#   Description:    A file that contains all Neural Network model class and/or objects.

import numpy as np
import matplotlib.pyplot as plt

from settings import *

# Model for predicting what tasks should be done
class Predictive_NN:
    def __init__(self, input_size, hidden_size, hidden2_size, output_size, learning_rate=0.01):
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
        d_hidden1 = error_hidden1 * self.sigmoid_derivative(self.a1)

        self.w3 += self.a2.T.dot(d_output) * self.learning_rate
        self.b3 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        self.w2 += self.a1.T.dot(d_hidden2) * self.learning_rate
        self.b2 += np.sum(d_hidden2, axis=0, keepdims=True) * self.learning_rate

        self.w1 += X.T.dot(d_hidden1) * self.learning_rate
        self.b1 += np.sum(d_hidden1, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=15000):
        printProgressBar(0, epochs, prefix='PROGRESS:', suffix='Complete', length=50)

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 100 == 0:
                self.training_loss.append(np.mean((y - output) ** 2))

            printProgressBar(epoch+1, epochs, prefix='PROGRESS:', suffix='Complete', length=50)
        
        print()
        return self.training_loss

def softmax(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)

    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones((1, 1, dim))
        self.beta = np.ones((1, 1, dim))

    def forward(self, x):
        self.mean = x.mean(-1, keepdims=True)
        self.var = x.var(-1, keepdims=True)
        self.norm = (x - self.mean) / np.sqrt(self.var + self.eps)

        return self.gamma * self.norm + self.beta
    
    def backward(self, dout):
        N, L, D = dout.shape
        x_mu = self.norm
        std_inv = 1. / np.sqrt(self.var + self.eps)

        dx_norm = dout * self.gamma
        dvar = np.sum(dx_norm * (x_mu * -0.5) * std_inv**3, axis=1, keepdims=True)
        dmean = np.sum(-dx_norm * std_inv, axis=1, keepdims=True) + dvar * np.mean(-2. * x_mu, axis=-1, keepdims=True)

        dx = dx_norm * std_inv + dvar * 2 * x_mu / D + dmean / D
        self.dgamma = np.sum(dout * self.norm, axis=(0, 1), keepdims=True)
        self.dbeta = np.sum(dout, axis=(0, 1), keepdims=True)

        return dx
    
class MultiHeadSelfAttention:
    def __init__(self, embed_size, heads):
        self.heads = heads
        self.embed_size = embed_size
        self.head_dim = embed_size // heads

        self.wq = np.random.randn(embed_size, embed_size) / np.sqrt(embed_size)
        self.wk = np.random.randn(embed_size, embed_size) / np.sqrt(embed_size)
        self.wv = np.random.randn(embed_size, embed_size) / np.sqrt(embed_size)
        self.wo = np.random.randn(embed_size, embed_size) / np.sqrt(embed_size)

    def split_heads(self, x):
        B, T, D = x.shape

        return x.reshape(B, T, self.heads, self.head_dim).transpose(0, 2, 1, 3)
    
    def combine_heads(self, x):
        B, H, T, D = x.shape

        return x.transpose(0, 2, 1, 3).reshape(B, T, H * D)
    
    def forward(self, x):
        self.x = x
        B, T, D = x.shape

        self.q = self.split_heads(x @ self.wq)
        self.k = self.split_heads(x @ self.wk)
        self.v = self.split_heads(x @ self.wv)

        scores = self.q @ self.k.transpose(0, 1, 3, 2) / np.sqrt(self.head_dim)
        self.attn = softmax(scores, axis=-1)
        self.context = self.attn @ self.v

        out = self.combine_heads(self.context) @ self.wo
        return out
    
    

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
