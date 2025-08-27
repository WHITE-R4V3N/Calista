#   Author:         Emma Gillespie
#   Date:           2025-06-14
#   Description:    This is the file that houses the model for the capstone project.
#                   This boasts a feedforward network that will use one-hot encoding
#                   to determine what the user is asking.

#----------------
#    IMPORTS    |
#----------------
import numpy as np
import matplotlib.pyplot as plt

import shlex
import subprocess

from common import *

#------------------------------------------------------------
#   A feedforward network with an additional hidden layer   |
#------------------------------------------------------------
class FeedForwardNetwork:
    def __init__(self, input_size, hidden_size, hidden2_size, output_size, learning_rate=0.01, debug=True):
        # Model Param's
        self.debug = debug
        
        self.w1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.w2 = np.random.randn(hidden_size, hidden2_size)
        self.b2 = np.zeros((1, hidden2_size))

        self.w3 = np.random.randn(hidden2_size, output_size)
        self.b3 = np.zeros((1, output_size))

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

        self.z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.sigmoid(self.z3)

        return self.a3
    
    def backward(self, X, y, output):
        error = y - output
        d_output = error * self.sigmoid_derivative(output)

        error_hidden = d_output.dot(self.w3.T)
        d_hidden2 = error_hidden * self.sigmoid_derivative(self.a2)
        d_hidden = d_hidden2 * self.sigmoid_derivative(self.a1)

        self.w3 += self.a2.T.dot(d_output) * self.learning_rate
        self.b3 += np.sum(d_output, axis=0, keepdims=True) * self.learning_rate

        self.w2 += self.a1.T.dot(d_hidden2) * self.learning_rate
        self.b2 += np.sum(d_hidden2, axis=0, keepdims=True) * self.learning_rate

        self.w1 += X.T.dot(d_hidden) * self.learning_rate
        self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.learning_rate

    def train(self, X, y, epochs=1200):
        printProgressBar(0, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)
        self.loss_history = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            loss = np.mean((y - output) ** 2)
            self.loss_history.append(loss)

            printProgressBar(epoch + 1, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)

        if self.debug == True:
            self.plot_loss()
            self.show_hidden_weights()
        
        return loss

    def predict(self, X):
        output = self.forward(X)

        return np.argmax(output, axis=1)
    
    def plot_loss(self):
        # Show training loss over epochs
        plt.plot(self.loss_history)
        plt.title("Feed Forward Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def show_hidden_weights(self):
        # Show hidden weights for last forward pass
        if hasattr(self, 'w2'):
            plt.imshow(self.w2, cmap='viridis')
            plt.title("Hidden Weights")
            plt.xlabel("Hidden")
            plt.ylabel("Hidden 2")
            plt.colorbar()
            plt.show()
    
    def compute_confidence_score(self, pred):
        return max(max(pred/len(pred)))
    
class Transformer:
    def __init__(self, input_size, output_size, d_model=64, max_length=32, learning_rate=0.001, debug=True):
        # Model Param's
        self.d_model = d_model
        self.output_size = output_size
        self.input_size = input_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.debug = debug

        # Word embeddings (vocab_size, d_model)
        self.embeddings = np.random.randn(input_size, d_model) * self.learning_rate

        # Positional encodings (max_length, d_model)
        self.positional_encoding = self._init_positional_encoding(max_length, d_model)

        # Attention projection weights
        self.w_q = np.random.randn(d_model, d_model) * self.learning_rate
        self.w_k = np.random.randn(d_model, d_model) * self.learning_rate
        self.w_v = np.random.randn(d_model, d_model) * self.learning_rate
        self.w_o = np.random.randn(d_model, d_model) * self.learning_rate

        # Final output projection (d_model, vocab_size)
        self.fc_out = np.random.randn(d_model, output_size) * self.learning_rate

    def _init_positional_encoding(self, max_length, d_model):
        # Sinusoidal positional encoding
        pos = np.arange(max_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

        pos_encoding = pos * angle_rates
        pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])

        return pos_encoding

    def softmax(self, x):
        # Prevent overflow
        x = np.clip(x, -500, 500)

        # Standard softmax function
        e_x = np.exp(x - np.max(x))
        sum_e_x = np.sum(e_x, axis=-1, keepdims=True)

        # Avoid division by zero (this is to fix a runtime warning)
        sum_e_x[sum_e_x == 0] = 1e-9
        return e_x / sum_e_x
    
    def scaled_dot_product(self, Q, K, V):
        # Compute attention weights and output
        d_k = Q.shape[-1]
        scores = np.dot(Q, K.T) / np.sqrt(d_k)
        weights = self.softmax(scores)

        return np.dot(weights, V), weights
    
    def forward(self, x):
        # Embedding lookup and apply the positional encoding
        self.x_ids = x
        self.x = np.array([self.embeddings[token] for token in x])
        self.x += self.positional_encoding[:len(x)]

        # Attention projections
        self.Q = self.x @ self.w_q
        self.K = self.x @ self.w_k
        self.V = self.x @ self.w_v

        # Compute attention output
        self.a_o, self.a_w = self.scaled_dot_product(self.Q, self.K, self.V)

        # Final projection
        self.a_op = self.a_o @ self.w_o

        # Mean pooling across sequence
        self.mean_pooled = self.a_op.mean(axis=0)

        # Output logits and probabilities
        self.logits = self.mean_pooled @ self.fc_out
        self.o = self.softmax(self.logits)

        #if self.debug:
        #    print(f'{GREY}[model-debug]{RESET} Logits: \n{self.logits}\n')
        #    print(f'{GREY}[model-debug]{RESET} Softmax output: \n{self.o}\n')

        return self.o
    
    def backward(self, y_ture):
        # Compute loss gradient
        error = self.o - y_ture

        # Backpropagation through the output layer
        d_fc_out = np.outer(self.mean_pooled, error)
        d_mean = error @ self.fc_out.T / self.a_op.shape[0]

        # Backpropagation through attention projection
        d_a_op = np.tile(d_mean, (self.a_op.shape[0], 1))
        d_w_o = self.a_o.T @ d_a_op
        d_a_o = d_a_op @ self.w_o.T

        # Backpropagation through attention weights
        d_weights = d_a_o @ self.V.T
        d_V = self.a_w.T @ d_a_o
        d_scores = d_weights * self.a_w * (1 - self.a_w)
        d_K = (self.Q.T @ d_scores).T
        d_Q = d_scores @ self.K

        # Backpropagation to Q, K, V weights
        d_w_q = self.x.T @ d_Q
        d_w_k = self.x.T @ d_K
        d_w_v = self.x.T @ d_V

        # Clip gradients to avoid explosion
        for grad in [d_fc_out, d_w_o, d_w_q, d_w_k, d_w_v]:
            np.clip(grad, -1.0, 1.0, out=grad)

        # Gradient descent updates
        self.fc_out -= self.learning_rate * d_fc_out
        self.w_o -= self.learning_rate * d_w_o
        self.w_q -= self.learning_rate * d_w_q
        self.w_k -= self.learning_rate * d_w_k
        self.w_v -= self.learning_rate * d_w_v

    def train(self, prompts, targets, epochs=1000):
        # Training the network by going through the backpropagation of the network
        printProgressBar(0, epochs, prefix='Transformer Training:', suffix='Complete', length=50)

        self.loss_history = []

        for epoch in range(epochs):
            loss_epoch = 0

            for i in range(len(prompts)):
                input_seq = prompts[i]
                target_seq = targets[i]
                out = self.forward(input_seq)
                loss = -np.sum(target_seq * np.log(out + 1e-9)) # Cross Entropy Loss
                loss_epoch += loss

                self.backward(target_seq)
            
            mean_loss = loss_epoch / len(prompts)
            self.loss_history.append(mean_loss)

            printProgressBar(epoch + 1, epochs, prefix='Transformer Training:', suffix='Complete', length=50)
        
        if self.debug == True:
            self.plot_loss()
            self.show_attention_weights()

        return mean_loss
    
    def predict(self, x):
        # Predict the token probabilities
        return self.forward(x)
    
    def generate(self, prompt, tokenizer, max_length=10):
        # Autoregressive generation of tokens from prompt
        context = tokenizer.encode(prompt, self.max_length)
        generated = [tokenizer.word2idx['<sos>']]
        #tokens = tokenizer.encode(prompt, self.max_length)

        for _ in range(max_length):
            input_seq = context + generated[-self.max_length:]
            input_seq = input_seq[-self.max_length:]
            out = self.forward(input_seq)

            next_token = np.argmax(out)

            if next_token == tokenizer.word2idx['<eos>']:
                break
            generated.append(next_token)

        print(f'\n{GREY}[model-generate]{RESET}\n{generated}')
        return tokenizer.decode(generated[1:])
    
    def generate_with_sampling(self, prompt, tokenizer, k=5, temperature=1.0, max_length=10):
        context = tokenizer.encode(prompt, self.max_length)
        generated  =[tokenizer.word2idx['<sos>']]
        #tokens = tokenizer.encode(prompt, self.max_length)

        for _ in range(max_length):
            input_seq = context + generated[-self.max_length:]
            input_seq = input_seq[-self.max_length:]
            
            out = self.forward(input_seq)
            out = np.log(out + 1e-9) / temperature

            probs = np.exp(out) / np.sum(np.exp(out))

            top_k_indices = probs.argsort()[-k:][::-1]
            top_k_probs = probs[top_k_indices]
            top_k_probs /= np.sum(top_k_probs)

            next_token = np.random.choice(top_k_indices, p=top_k_probs)

            if next_token == tokenizer.word2idx['<eos>']:
                break
            generated.append(next_token)

        print(f'\n{GREY}[model-generate-sampling]{RESET}\n{generated}')
        return tokenizer.decode(generated[1:])
    
    def batch_generate_from_file(self, file_path, tokenizer):
        # Add the logs.write_logs() after the call and save the results to the logs so that we know what happened
        with open(file_path, 'r') as f:
            prompts = f.read().splitlines()

        results = {}
        for prompt in prompts:
            command = self.generate_with_sampling(prompt, tokenizer, k=5, temperature=0.8)
            results[prompt] = command
        return results
    
    def preview_execution(self, command_str):
        try:
            parts = shlex.split(command_str)
            print(f'{YELLOW}[-]{RESET} Preview (sandboxed): {command_str}\n')
            result = subprocess.run(parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            print(f'{result.stdout.decode()[:500]}\n')
            return True
        except Exception as e:
            print(f'{RED}[x]{RESET}Execution Error: {e}\n')
            return False
    
    def plot_loss(self):
        # Show training loss over epochs
        plt.plot(self.loss_history)
        plt.title("Transformer Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

    def show_attention_weights(self):
        # Show attention weights for last forward pass
        if hasattr(self, 'a_w'):
            plt.imshow(self.a_w, cmap='viridis')
            plt.title("Attention Weights")
            plt.xlabel("Key")
            plt.ylabel("Query")
            plt.colorbar()
            plt.show()

    def compute_confidence_score(self, pred):
        return max(pred)
