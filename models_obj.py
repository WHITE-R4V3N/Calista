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

class Transformer:
    def __init__(self, vocab_size, d_model, seq_len, num_heads, learning_rate):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.learning_rate = learning_rate

        self.embedding = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)
        self.positional_encoding = self.create_positional_encoding(seq_len, d_model)

        self.wq = np.random.randn(num_heads, d_model, d_model // num_heads)
        self.wk = np.random.randn(num_heads, d_model, d_model // num_heads)
        self.wv = np.random.randn(num_heads, d_model, d_model // num_heads)

        self.wo = np.random.randn(num_heads * (d_model // num_heads), d_model)
        self.w1 = np.random.randn(d_model, 4 * d_model)
        self.w2 = np.random.randn(4 * d_model, d_model)

    def create_positional_encoding(self, seq_len, d_model):
        pos_enc = np.zeros((seq_len, d_model))

        for pos in range(seq_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos /(10000 ** (i / d_model)))

                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        
        return pos_enc
    
    def embed_input(self, X):
        return self.embedding[X] + self.positional_encoding[: len(X)]
    
    def multi_head_attention(self, X):
        print(X)
        print(X.shape)
        batch_size, seq_len, d_model = X.shape

        q = np.dot(X, self.wq)
        k = np.dot(X, self.wk)
        v = np.dot(X, self.wv)

        scores = np.einsum('nhqd,nhkd->nhqk', q, k) / np.sqrt(d_model)
        a_w = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        context = np.einsum('nhqk,nhvd->nhqd', a_w, v)
        context = context.reshape(batch_size, seq_len, -1)
        o = np.dot(context, self.wo)

        return o, a_w
    
    def feed_forward(self, X):
        return np.dot(np.maximum(0, np.dot(X, self.w1)), self.w2)
    
    def forward(self, X):
        X = self.embed_input(X)
        a_o, a_w = self.multi_head_attention(X)

        X = a_o + X
        X = self.feed_forward(X) + X

        return X, a_w
    
    def backward(self, X, grad_output):
        grad_w2 = np.dot(self.feed_forward(X).T, grad_output)
        grad_hidden = np.dot(grad_output, self.w2.T)
        grad_w1 = np.dot(X.T, np.maximum(0, grad_hidden))
        grad_a = np.dot(grad_output, self.wq.T)

        self.w2 -= self.learning_rate * grad_w2
        self.w1 -= self.learning_rate * grad_w1
        self.wq -= self.learning_rate * grad_a
        self.wk -= self.learning_rate * grad_a
        self.wv -= self.learning_rate * grad_a
        self.wo -= self.learning_rate * grad_a

    def generate_command(self, seed_input, max_length=20):
        command = [seed_input]

        for _ in range(max_length - 1):
            input_seq = np.array(command).reshape(1, len(command), self.d_model)
            output, _ = self.forward(input_seq)
            
            next_token = np.argmax(output[:, -1, :])
            command.append(next_token)

            if next_token == 0: # Assuming 0 is the end token for now. We will need to change and update this to a different token value.
                break

        return command
    
    def train(self, X, y, epochs=15000):
        printProgressBar(0, epochs, prefix='PROGRESS:', suffix='Complete', length=50)

        for epoch in range(epochs):
            loss_array = []

            for i in range(len(X)):
                output, _ = self.forward(X)
                loss = np.mean((output - y) ** 2)   # Mean squared Error Loss

                grad_output = 2 * (output - y) / y.size
                self.backward(X, grad_output)
                loss_array.append[loss/len(X)]
            
            printProgressBar(epoch+1, epochs, prefix='PROGRESS:', suffix='Complete', length=50)

        return loss_array

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
