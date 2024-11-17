#   Description:    A file that contains all Neural Network model class and/or objects.

import numpy as np

class Base_Model:
    def __init__(self, model, tokenizer, dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.X = []
        self.X_min = 0
        self.X_max = 0
        self.y = []
        self.labels = []
        self.labels_index = []

    def create_X_y_training(self, algorithm_cipher):
        for ci_text in algorithm_cipher:
            self.labels_index.append(algorithm_cipher[ci_text])

        self.labels_index = list(dict.fromkeys(self.labels_index)) # Create the labels used by the cryptographic AI model

        for ciphered_text in algorithm_cipher:
            self.X.append(self.tokenizer.char_tokenize(ciphered_text))
            self.X = self.tokenizer.pad_input(self.X)
            self.labels.append(algorithm_cipher[ciphered_text])

        for label in self.labels:
            self.y.append(self.labels_index.index(label))

        self.y = np.eye(len(self.labels_index))[self.y] # one-hot encoding
        self.X = np.array(self.X) # Normalize x before using. Causes overflow otherwise

        # Here is where we normalize the X so there is no overflow in the data.
        self.X_min = self.X.min(axis=0)
        self.X_max = self.X.max(axis=0)
        self.X = (self.X - self.X_min) / (self.X_max - self.X_min)

class Simplified_NN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def forward(self, X):
        self.z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.z2)

        return self.a2
    
    def backward(self, X, y, output):
        error = y - output
        d_output = error * self.sigmoid_derivative(output)

        error_hidden = d_output.dot(self.w2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        self.w2 += self.a1.T.dot(d_output) * self.learning_rate
        self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate
        self.w1 += X.T.dot(d_hidden) * self.learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f'Epoch: {epoch} has a loss of {loss}')
        
        return loss

    def predict(self, X):
        output = self.forward(X)

        return np.argmax(output, axis=1)