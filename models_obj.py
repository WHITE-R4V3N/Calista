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
    def __init__(self, embed_size, num_heads):
        self.embed_size = embed_size
        self.heads = num_heads
        self.head_dim = embed_size // num_heads

        assert self.head_dim * num_heads == embed_size, "Embedding size must be evenly divisable by heads."

        self.w_q = np.random.randn(embed_size, embed_size)
        self.w_k = np.random.randn(embed_size, embed_size)
        self.w_v = np.random.randn(embed_size, embed_size)
        self.w_o = np.random.randn(embed_size, embed_size)

    def forward(self, q, k, v):
        self.q = np.dot(q, self.w_q)
        self.k = np.dot(k, self.w_k)
        self.v = np.dot(v, self.w_v)

        scores = np.dot(self.q, self.k.T) / np.sqrt(self.head_dim)
        self.a_w = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        self.o = np.dot(self.a_w, self.v)

        return np.dot(self.o, self.w_o)

    def backward(self, d_out):
        d_v = np.dot(d_out, self.w_o.T)
        d_w_o = np.dot(self.o.T, d_out)

        d_a = np.dot(d_v, self.v.T)
        d_scores = d_a * self.a_w * (1 - self.a_w)
        
        d_q = np.dot(d_scores, self.k)
        d_k = np.dot(d_scores.T, self.q)
        d_v = np.dot(self.a_w.T, d_v)

        self.w_q -= 0.001 * np.dot(self.q.T, d_q)
        self.w_k -= 0.001 * np.dot(self.k.T, d_k)
        self.w_v -= 0.001 * np.dot(self.v.T, d_v)
        self.w_o -= 0.001 * d_w_o
    
class TransformerBlock:
    def __init__(self, embed_size, num_heads, hidden_dim):
        self.a = MultiHeadAttention(embed_size, num_heads)

        self.w1 = np.random.randn(embed_size, hidden_dim)
        self.w2 = np.random.randn(hidden_dim, embed_size)

    def forward(self, X):
        self.a_o = self.a.forward(X, X, X)
        self.ff_o = np.dot(np.maximum(0, np.dot(self.a_o, self.w1)), self.w2)

        return self.ff_o

    def backward(self, d_out):
        d_ff = np.dot(d_out, self.w2.T)

        d_w2 = np.dot(np.maximum(0, self.a_o).T, d_out)
        d_w1 = np.dot(self.a_o.T, d_ff)

        self.w1 -= 0.001 * d_w1
        self.w2 -= 0.001 * d_w2

        self.a.backward(d_ff)

class Transformer:
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, hidden_dim):
        self.embedding = EmbeddingLayer(vocab_size, embed_size)
        self.pos_encoding = PositionalEncoding(embed_size)

        self.layers = [TransformerBlock(embed_size, num_heads, hidden_dim) for _ in range(num_layers)]

        self.fc_o = np.random.randn(embed_size, vocab_size)

    def forward(self, X):
        X = self.embedding.forward(X)
        X = self.pos_encoding.forward(X)

        for layer in self.layers:
            X = layer.forward(X)
        
        return np.dot(X, self.fc_o)

    def backward(self, d_out):
        d_fc = np.dot(d_out, self.fc_o.T)
        d_w_fc = np.dot(d_fc.T, d_out)

        self.fc_o -= 0.001 * d_w_fc

        for layer in reversed(self.layers):
            layer.backward(d_fc)
        self.embedding.backward(d_fc)

    def generate(self, seed, length=10):
        for _ in range(length):
            pred = self.forward(seed)
            n_token = np.argmax(pred, axis=-1)

            print(f'Seed: \n{seed}\nLength: {len(seed)}')
            print(f'[n_token]: \n{list(n_token)}\nLength: {len(list(n_token))}')

            seed = np.append(seed, list(n_token), axis=0)
        
        return seed

    def train(self, X, y, epochs=15000):
        printProgressBar(0, epochs, prefix='PROGRESS:', suffix='Complete', length=50)
        
        total_loss = []
        for epoch in range(epochs):
            pred = self.forward(X)

            loss = -np.sum(y, *np.log(pred)) # Cross entropy loss
            self.backward(pred - y)
            
            printProgressBar(epoch+1, epochs, prefix='PROGRESS:', suffix='Complete', length=50)

        return total_loss

class EmbeddingLayer:
    def __init__(self, vocab_size, embed_size):
        self.weights = np.random.randn(vocab_size, embed_size)

    def forward(self, X):
        self.X = X

        return self.weights[X]

    def backward(self, d_out):
        self.weights[self.X] -= 0.001 * d_out

class PositionalEncoding:
    def __init__(self, embed_size, max_len=10000):
        self.encoding = np.zeros((max_len, embed_size))

        for pos in range(max_len):
            for i in range(0, embed_size, 2):
                self.encoding[pos, i] = np.sin(pos / (10000 ** (i / embed_size)))

                if i + 1 < embed_size:
                    self.encoding[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / embed_size)))

    def forward(self, X):
        print(f'Positional X: {len(X)}')
        print(f'Encoding: {len(self.encoding)}')
        print(f'Encoding X shape[0]: {len(self.encoding[:X.shape[0]])}')

        return X + self.encoding[:X.shape[0]]

def generate_text(transformer, seed, length, vocab_size):
    generated = list(seed)

    print(f'Generated {generated}')

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
