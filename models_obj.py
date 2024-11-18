#   Description:    A file that contains all Neural Network model class and/or objects.

import numpy as np
import matplotlib.pyplot as plt

from settings import *

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
        printProgressBar(0, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)

            printProgressBar(epoch + 1, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
        return loss

    def predict(self, X):
        output = self.forward(X)

        return np.argmax(output, axis=1)
    
    def visualize(self, input_data):
        input_values = input_data
        hidden_values = np.tanh(np.dot(input_data, self.w1) + self.b1)
        output_values = self.sigmoid(np.dot(hidden_values, self.w2) + self.b2)

        Visualize_NN(input_values[0], hidden_values[0], output_values[0], self.w1, self.w2)
    
def Visualize_NN(input_values, hidden_values, output_values, w1, w2):
    # Parameters for layout
    max_layer_size = max(len(input_values), len(hidden_values), len(output_values))
    layer_spacing = 100
    node_spacing = 30 / max_layer_size
    node_radius = 2

    # Number of nodes in each layer
    input_size = len(input_values)
    hidden_size = len(hidden_values)
    output_size = len(output_values)

    # Create figure
    fig_width = 12
    fig_height = min(18, max_layer_size * node_spacing)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')
    ax.set_aspect('equal')

    # Adjust limits to prevent stretching
    #x_margin = layer_spacing * 0.5
    #y_margin = node_spacing * max_layer_size * 0.5
    #ax.set_xlim(-x_margin, 2 * layer_spacing + x_margin)
    #ax.set_ylim(-y_margin, y_margin)

    def draw_layer(values, x_position, layer_size, color_map):
        y_positions = np.linspace(-layer_size / 2, layer_size / 2, layer_size)
        circles = []

        for i, y in enumerate(y_positions):
            color = color_map(values[i])
            circle = plt.Circle((x_position, y), node_radius, color=color, ec='black', lw=0.5)
            circles.append(circle)

        return y_positions, circles

    def color_map(value):
        if value > 0:
            intensity = min(1, value)
            return (0, intensity, 0) # RGB values
        else:
            return (0.8, 0.8, 0.8) # RGB values
        
    # Draw Layers
    input_positions, input_circles = draw_layer(input_values, 0, input_size, color_map)
    hidden_positions, hidden_circles = draw_layer(hidden_values, layer_spacing, hidden_size, color_map)
    output_positions, output_circles = draw_layer(output_values, 2 * layer_spacing, output_size, color_map)

    def draw_connections(weights, x_start, y_start, x_end, y_end):
        for i, y1 in enumerate(y_start):
            for j, y2 in enumerate(y_end):
                weight = weights[i, j]
                color = (0.5, 0.5, min(1, abs(weight))) # Color blue for connection
                lw = max(0.1, abs(weight))          # Line width based on weight magnitude
                ax.plot([x_start, x_end], [y1, y2], color=color, lw=lw, zorder=1)

    draw_connections(w1, 0, input_positions, layer_spacing, hidden_positions)
    draw_connections(w2, layer_spacing, hidden_positions, (2 * layer_spacing), output_positions)

    for circle in input_circles + hidden_circles + output_circles:
        ax.add_artist(circle)

    plt.show()