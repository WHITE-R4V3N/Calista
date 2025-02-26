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

class MultiHeadAttention:
    def __init__(self, embed_size, num_heads, learning_rate):
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.learning_rate = learning_rate

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size must be divisible by number of heads."

        self.w_q = np.random.randn(embed_size, embed_size) * 0.01
        self.w_k = np.random.randn(embed_size, embed_size) * 0.01
        self.w_v = np.random.randn(embed_size, embed_size) * 0.01
        self.w_o = np.random.randn(embed_size, embed_size) * 0.01

    def forward(self, q, k, v):
        Q = np.dot(q, self.w_q)
        K = np.dot(k, self.w_k)
        V = np.dot(v, self.w_v)

        Q = Q.reshape(Q.shape[0], -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(K.shape[0], -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(V.shape[0], -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        attention = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention /= attention.sum(axis=-1, keepdims=True)
        output = np.matmul(attention, V)

        output = output.transpose(0, 2, 1, 3).reshape(output.shape[0], -1, self.embed_size)

        return np.dot(output, self.w_o), attention
    
    def backward(self, d_output, learning_rate):
        self.w_q -= learning_rate * d_output
        self.w_k -= learning_rate * d_output
        self.w_v -= learning_rate * d_output
        self.w_o -= learning_rate * d_output
    
class TransformerBlock:
    def __init__(self, embed_size, num_heads, feedforward_dim):
        self.attention = MultiHeadAttention(embed_size, num_heads, 0.001)
        self.feedforward = np.random.randn(embed_size, feedforward_dim) * 0.01
        self.ff_o = np.random.randn(feedforward_dim, embed_size) * 0.01

    def forward(self, X):
        a_o, a_w = self.attention.forward(X, X, X)
        X = X + a_o

        ff_o = np.maximum(0, np.dot(X, self.feedforward))
        ff_o = np.dot(ff_o, self.ff_o)
        X = X + ff_o

        return X, a_w
    
    def backward(self, d_output, learning_rate):
        self.attention.backward(d_output, learning_rate)
        self.feedforward -= learning_rate * d_output
        self.ff_o -= learning_rate * d_output

class Transformer:
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, feedforward_dim):
        self.embed_size = embed_size
        self.embedding = EmbeddingLayer(vocab_size, embed_size)
        self.positional_encoding = np.random.randn(5865, embed_size) * 0.01

        self.layers = [
            TransformerBlock(embed_size, num_heads, feedforward_dim)
            for _ in range(num_layers)
        ]

    def forward(self, X):
        X = self.embedding.forward(X) + self.positional_encoding[:X.shape[1]]

        attention_weights_list = []

        for layer in self.layers:
            X, attention_weights = layer.forward(X)
            attention_weights_list.append(attention_weights)

        return X, attention_weights_list
    
    def backward(self, d_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(d_output, learning_rate)

        self.embedding.backward(grad_output, learning_rate)

    def cross_entropy_loss(self, predictions, targets):
        predictions = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
        predictions /= predictions.sum(axis=-1, keepdims=True)

        if predictions.shape[0] != targets.shape[0]:
            raise ValueError(f'Shape Mismatch\nprediction shape: {predictions.shape} | targets shape: {targets.shape}')

        print(f'Prediction Shape: {predictions.shape}')
        print(f'Target shape: {targets.shape}')

        loss = -np.sum(np.log(predictions[np.arange(len(targets)), targets] + 1e-9)) / len(targets)
        return loss

    def train(self, X, y, learning_rate, vocab_size, epochs=15000):
        printProgressBar(0, epochs, prefix='PROGRESS:', suffix='Complete', length=50)
        
        total_loss = []
        for epoch in range(epochs):

            # Forward pass
            logits, _ = self.forward(X)

            print(f'Logits shape: {logits.shape}\n')

            loss  = self.cross_entropy_loss(logits[0], y)
            total_loss.append(loss)

            dummy_grad = logits - np.eye(vocab_size)[y]
            self.backward(dummy_grad, learning_rate)
            
            printProgressBar(epoch+1, epochs, prefix='PROGRESS:', suffix='Complete', length=50)

        return total_loss

class EmbeddingLayer:
    def __init__(self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.weights = np.random.randn(vocab_size, embed_size) * 0.01

    def forward(self, X):
        return self.weights[X]

    def backward(self, grad_output, learning_rate):
        self.weights -= learning_rate * grad_output

def generate_text(transformer, seed, length, vocab_size):
    generated = list(seed)

    print(f'Length of generated: {len(generated)}\nGenerate')

    for _ in range(length):
        logits, _ = transformer.forward(np.array([generated]))
        next_token = np.argmax(logits[0, -1])
        generated.append(next_token)
    
    return generated

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
