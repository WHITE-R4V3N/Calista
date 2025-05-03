# Author:       Emma Gillespie
# Date:         2025-04-30
# Description:  This is where the main transformer model will be. It integrates the transformer model
#               architecture capable of generating shell commands based on task descriptions. (This can
#               be generated based on a predictive feed forward model).

import numpy as np
from settings import *
from tokenizer import Tokenizer
from layers import EncoderLayer, DecoderLayer

class Transformer:
    def __init__(self, d_model, num_heads, d_ff, src_vocab_size, tgt_vocab_size, max_seq_len):
        self.encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
        self.decoder_layer = DecoderLayer(d_model, num_heads, d_ff)

        self.src_embedding = np.random.randn(src_vocab_size, d_model) / np.sqrt(d_model)
        self.tgt_embedding = np.random.randn(tgt_vocab_size, d_model) / np.sqrt(d_model)

        self.pos_embedding = np.random.randn(max_seq_len, d_model) / np.sqrt(d_model)

        self.output_projection = np.random.randn(d_model, tgt_vocab_size) / np.sqrt(d_model)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        batch_size, src_len = src.shape
        _, tgt_len = tgt.shape

        src_embed = self.src_embedding[batch_size] + self.pos_embedding[:src_len]
        tgt_embed = self.tgt_embedding[batch_size] + self.pos_embedding[:tgt_len]

        enc_output = self.encoder_layer.forward(src_embed, src_mask)
        dec_output = self.decoder_layer.forward(tgt_embed, enc_output, src_mask, tgt_mask)

        logits = np.matmul(dec_output, self.output_projection)
        return logits
    
    def backward(self, d_logits):
        d_dec_output = np.matmul(d_logits, self.output_projection)
        d_output_proj = np.matmul(self.decoder_layer.norm3.out.reshape(-1, d_logits.shape[-1]).T, d_logits.reshape(-1, d_logits.shape[-1]))

        decoder_grads = self.decoder_layer.backward(d_dec_output)
        encoder_grads = self.encoder_layer.backward(decoder_grads['self_a'][0])

        return decoder_grads, encoder_grads, d_output_proj
    
def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy_loss(logits, targets):
    probs = softmax(logits)
    log_probs = -np.log(probs[np.arange(len(targets.flatten())), targets.flatten()] + 1e-9)
    loss = np.mean(log_probs)

    d_logits = probs
    d_logits[np.arange(len(targets.flatten())), targets.flatten()] -= 1
    d_logits /= targets.shape[0]

    return loss, d_logits

def train(model, data, targets, epochs=10, lr=1e-3):
    printProgressBar(0, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for epoch in range(epochs):
        logits = model.forward(data, targets)
        loss, d_logits = cross_entropy_loss(logits.reshape(-1, logits.shape[-1]), targets)

        grads = model.backward(d_logits.reshape(logits.shape))

        printProgressBar(epoch + 1, epochs, prefix = 'Progress:', suffix = 'Complete', length = 50)
        print(f"Epoch: {epoch + 1}, Loss: {loss:.4f}")