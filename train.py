import numpy as np
import json

def cross_entropy_loss(predictions, targets):
    """Computes the loss using cross-entropy."""
    predictions = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
    predictions /= predictions.sum(axis=-1, keepdims=True)
    
    loss = -np.sum(np.log(predictions[np.arange(len(targets)), targets] + 1e-9)) / len(targets)
    return loss

def train(transformer, data, epochs, learning_rate, vocab_size):
    for epoch in range(epochs):
        total_loss = 0
        for sample in data:
            input_tokens = np.array(sample["input"])
            target_tokens = np.array(sample["target"])
            
            # Forward pass
            logits, _ = transformer.forward(input_tokens.reshape(1, -1))
            
            # Compute loss
            loss = cross_entropy_loss(logits[0], target_tokens)
            total_loss += loss
            
            # Backpropagation
            dummy_grad = logits - np.eye(vocab_size)[target_tokens]  # Gradient of softmax loss
            transformer.backward(dummy_grad, learning_rate)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(data)}")

# Example usage
vocab_size = 100
embed_size = 32
num_heads = 4
num_layers = 2
feedforward_dim = 64
learning_rate = 0.001
epochs = 10

transformer = Transformer(vocab_size, embed_size, num_heads, num_layers, feedforward_dim)

# Load custom training data
with open("training_data.json", "r") as f:
    training_data = json.load(f)

train(transformer, training_data, epochs, learning_rate, vocab_size)
