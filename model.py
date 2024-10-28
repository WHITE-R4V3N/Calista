# Author:       Emma Gillespie
# Date:         2024-03-20
# Description:  The AI model that gets trained and loaded. Eventually the model will be saved and loaded 
#               instead using this script after the first time being ran and any time its closed. Allowing
#               the user to provide feedback that will influence the AI model.

#----------------
#    IMPORTS    |
#----------------

import numpy as np

from json_data_parse import *
from settings import *


from settings import *
from json_data_parse import *

#-------------------------------------
#   The model and training the AI    |
#-------------------------------------

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, hidden_size_2, output_size):
        # Set initial weights and biases
        self.weights1 = np.random.rand(input_size, hidden_size)
        self.weights2 = np.random.rand(hidden_size, hidden_size_2)
        self.weights3 = np.random.rand(hidden_size_2, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, hidden_size_2))
        self.bias3 = np.zeros((1, output_size))

        # Add another for a second hidden layers

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_prop(self, X):
        z1 = X.dot(self.weights1) + self.bias1
        a1 = self.sigmoid(z1)
        z2 = a1.dot(self.weights2) + self.bias2
        a2 = self.sigmoid(z2)
        z3 = a2.dot(self.weights3) + self.bias3
        a3 = self.sigmoid(z3)

        return a1, a2, a3
    
    def backwards_prop(self, X, y, learning_rate):
        m = len(X)

        a1, a2, a3 = self.forward_prop(X)

        # Calculate the error
        delta3 = (a3 - y) * self.sigmoid_derivative(a3)
        delta2 = (delta3.dot(self.weights3.T)) * self.sigmoid_derivative(a2)
        delta1 = (delta2.dot(self.weights2.T)) * self.sigmoid_derivative(a1)

        # Update weights and biases
        self.weights3 -= learning_rate * a2.T.dot(delta3) / m
        self.weights2 -= learning_rate * a1.T.dot(delta2) / m
        self.weights1 -= learning_rate * X.T.dot(delta1) / m
        self.bias3 -= learning_rate * delta3.sum(axis=0) / m
        self.bias2 -= learning_rate * delta2.sum(axis=0) / m
        self.bias1 -= learning_rate * delta1.sum(axis=0) / m

        return a1, a2, a3

    def predict(self, X):
        _, _, a3 = self.forward_prop(X)
        return a3
    
# def create_model (input_size, hidden_size, hidden_size_2, output_size)
# returns { 'calista' : [model, X, y], 'recon_model' : [model, X, y], ...}

# Define network Parameters
input_size = len(X[0])
#input_size = len(corpus)
# 54
hidden_size = 20
# hidden_size = 15
hidden_size_2 = 10
# hidden_size_2 = 15
# Possibly add a 3rd hidden layer to the network
output_size = len(y[0])
#output_size = 5
learning_rate = 0.5

model = NeuralNetwork(input_size, hidden_size, hidden_size_2, output_size)

#------------------------
#   Train the network   |
#------------------------


# def train model (model, X, y, learning_rate, epochs)
epochs = 5000

print('\nTraining the AI:')
printProgressBar(0, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)
for epoch in range(epochs):
    a1, a2, a3 = model.backwards_prop(X, y, learning_rate)
    #print(f'A1: {a1}')
    #print(f'A2: {a2}')
    #print(f'A3: {a3}')
    printProgressBar(epoch + 1, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)
