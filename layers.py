# Author:       Emma Gillespie
# Date:         2025-04-30
# Description:  File containing all of the models layer specific functions.

import numpy as np

from settings import *

class Embedding:
    def __init__(self, vocab_size, embed_size):
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.w = np.random.randn(vocab_size, embed_size) / np.sqrt(embed_size)

    def forward(self, X):
        self.input = X

        return self.w[X]
    
    def backward(self, d_out):
        grad = np.zeros_like(self.w)

        for i, row in enumerate(self.input):
            for j, idx in enumerate(row):
                grad[idx] += d_out[i][j]

        return grad
    
class PositionalEncoding:
    def __init__(self, max_len, d_model):
        self.encoding = np.zeros((max_len, d_model))
        pos = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        self.encoding[:, 0::2] = np.sin(pos * div_term)
        self.encoding[:, 1::2] = np.cos(pos * div_term)

    def forward(self, X):
        return X + self.encoding[:X.shape[1]]
    
def softmax(X):
    exp = np.exp(X - np.max(X, axis=-1, keepdims=True))

    return exp / np.sum(exp, axis=-1, keepdims=True)

class ScaledDotProductAttention:
    def __init__(self):
        self.q = None
        self.k = None
        self.v = None

        self.aw = None
        self.mask = None

    def forward(self, q, k, v, mask=None):
        self.q, self.k, self.v, = q, k, v
        self.mask = mask 
        d_k = q.shape[-1]

        scores = np.matmul(q, k.transpose(0, 2, 1)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        self.aw = softmax(scores)
        output = np.matmul(self.aw, v)

        return output
    
    def backward(self, d_out):
        d_attn = np.matmul(d_out, self.v.Transpose(0, 2, 1))
        d_v = np.matmul(self.aw.transpose(0, 2, 1), d_out)

        d_scores = d_attn * self.aw * (1 - self.aw)

        dq = np.matmul(d_scores, self.k) / np.sqrt(self.q.shape[-1])
        dk = np.matmul(d_scores.transpose(0, 2, 1), self.q) / np.sqrt(self.q.shape[-1])

        return dq, dk, d_v
    
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.wq = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.wk = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.wv = np.random.randn(d_model, d_model) / np.sqrt(d_model)
        self.wo = np.random.randn(d_model, d_model) / np.sqrt(d_model)

        self.a = ScaledDotProductAttention()

    def split_heads(self, X):
        batch_size, seq_len, d_model = X.shape
        X = X.reshape(batch_size, seq_len, self.num_heads, self.d_k)

        return X.transpose(0, 2, 1, 3)
    
    def combine_heads(self, X):
        batch_size, heads, seq_len, d_k  = X.shape
        X = X.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, heads * d_k)

        return X
    
    def forward(self, q, k, v, mask=None):
        self.q_in, self.k_in, self.v_in = q, k, v

        q_p = np.matmul(q, self.wq)
        k_p = np.matmul(k, self.wk)
        v_p = np.matmul(v, self.wv)

        q_heads = self.split_heads(q_p)
        k_heads = self.split_heads(k_p)
        v_heads = self.split_heads(v_p)

        context = self.a.forward(q_heads, k_heads, v_heads, mask)
        context_combined = self.combine_heads(context)

        self.context = context
        output = np.matmul(context_combined, self.wo)

        return output
    
    def backward(self, d_out):
        d_context_combined = np.matmul(d_out, self.wo.T)
        d_context = d_context_combined.reshape(
            self.context.shape[0], self.num_heads, self.context.shape[2], self.d_k
        ).transpose(0, 2, 1, 3)

        dq, dk, dv = self.a.backward(d_context)

        dq = dq.transpose(0, 2, 1, 3).reshape(self.q_in.shape)
        dk = dk.transpose(0, 2, 1, 3).reshape(self.k_in.shape)
        dv = dv.transpose(0, 2, 1, 3).reshape(self.v_in.shape)

        dwq = np.matmul(self.q_in.reshape(-1, self.q_in.shape[-1]).T, dq.reshape(-1, dq.shape[-1]))
        dwk = np.matmul(self.k_in.reshape(-1, self.k_in.shape[-1]).T, dk.reshape(-1, dk.shape[-1]))
        dwv = np.matmul(self.v_in.reshape(-1, self.v_in.shape[-1]).T, dv.reshape(-1, dv.shape[-1]))
        dwo = np.matmul(self.context.transpose(0, 2, 1, 3).reshape(-1, self.d_k * self.num_heads).T, d_out.reshape(-1, d_out.shape[-1]))

        return dq, dk, dv, dwq, dwk, dwv, dwo
    
class LayerNormalization:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))

        self.eps = eps

    def forward(self, X):
        self.mean = np.mean(X, axis=-1, keepdims=True)
        self.var = np.var(X, axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.norm = (X - self.mean) / self.std
        self.out = self.gamma * self.norm + self.beta

        return self.out
    
    def backward(self, d_out):
        n = d_out.shape[-1]

        d_norm = d_out * self.gamma
        d_var = np.sum(d_norm * (self.out - self.beta) * -0.5 * self.std**-3, axis=-1, keepdims=True)
        d_mean = np.sum(d_norm * -1 / self.std, axis=-1, keepdims=True) + d_var * np.mean(-2 * (self.out - self.beta), axis=-1, keepdims=True)

        dx = d_norm / self.std + d_var * 2 * (self.out - self.beta) / n + d_mean / n
        d_gamma = np.sum(d_out * self.norm, axis=(0, 1), keepdims=True)
        d_beta = np.sum(d_out, axis=(0, 1), keepdims=True)

        return dx, d_gamma, d_beta
        
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.w1 = np.random.randn(d_model, d_ff) / np.sqrt(d_model)
        self.b1 = np.zeros((1, 1, d_ff))
        self.w2 = np.random.randn(d_ff, d_model) / np.sqrt(d_ff)
        self.b2 = np.zeros((1, 1, d_model))

    def relu(self, X):
        return np.maximum(0, X)
    
    def relu_backward(self, dout):
        return dout * (self.h1 > 0)
    
    def forward(self, X):
        self.X = X
        self.h1 = np.matmul(X, self.w1) + self.b1
        self.h2 = self.relu(self.h1)
        out = np.matmul(self.h2, self.w2) + self.b2

        return out
    
    def backward(self, d_out):
        d_h2 = np.matmul(d_out, self.w2.T)
        d_w2 = np.matmul(self.h2.reshape(-1, self.h2.shape[-1]).T, d_out.reshape(-1, d_out.shape[-1]))
        d_b2 = np.sum(d_out, axis=(0, 1), keepdims=True)

        d_h1 = self.relu_backward(d_h2)
        d_w1 = np.matmul(self.X.reshape(-1, self.X.shape[-1]).T, d_h1.reshape(-1, d_h1.shape[-1]))
        d_b1 = np.sum(d_h1, axis=(0, 1), keepdims=True)

        d_x = np.matmul(d_h1, self.w1.T)

        return d_x, d_w1, d_b1, d_w2, d_b2
    
class EncoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_a = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = LayerNormalization(d_model)

    def forward(self, x, mask=None):
        a_o = self.self_a.forward(x, x, x, mask)
        x = self.norm1.forward(x + a_o)

        ffn_o = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_o)

        return x
    
    def backward(self, d_out):
        d_ffn, d_gamma2, d_beta2 = self.norm2.backward(d_out)
        d_ffn_o, dw1, db1, dw2, db2 = self.ffn.backward(d_ffn)

        d_a_o, d_gamma1, d_beta1 = self.norm1.backward(d_ffn_o)
        dq, dk, dv, dwq, dwk, dwv, dwo = self.self_a.backward(d_a_o)

        return {
            "dq": dq, "dk": dk, "dv": dv,
            "dwq": dwq, "dwk": dwk, "dwv": dwv, "dwo": dwo,
            "dw1":dw1, "db1": db1, "dw2": dw2, "db2": db2,
            "d_gamma1": d_gamma1, "d_beta1": d_beta1,
            "d_gamma2": d_gamma2, "d_beta2": d_beta2,
        }

class DecoderLayer:
    def __init__(self, d_model, num_heads, d_ff):
        self.self_a = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalization(d_model)

        self.cross_a = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNormalization(d_model)

        self.ffn = FeedForward(d_model, d_ff)
        self.norm3 = LayerNormalization(d_model)

    def forward(self, x, enc_o, src_mask=None, tgt_mask=None):
        self_a_o = self.self_a.forward(x, x, x, tgt_mask)
        x = self.norm1.forward(x + self_a_o)

        cross_a_o = self.cross_a.forward(x, enc_o, enc_o, src_mask)
        x = self.norm2.forward(x + cross_a_o)

        ffn_o = self.ffn.forward(x)
        x = self.norm3.forward(x + ffn_o)

        return x
    
    def backward(self, d_o):
        d_ffn, d_gamma3, d_beta3 = self.norm3.backward(d_o)
        d_ffn_o, dw1, db1, dw2, db2 = self.ffn.backward(d_ffn)

        d_cross, d_gamma2, d_beta2 = self.norm2.backward(d_ffn_o)
        dq2, dk2, dv2, dwq2, dwk2, dwv2, dwo2 = self.self_a.backward(d_cross)

        d_self, d_gamma1, d_beta1 = self.norm1.backward(dq2)
        dq1, dk1, dv1, dwq1, dwk1, dwv1, dwo1 = self.self_a.backward(d_self)

        return {
            "dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2,
            "d_gamma1": d_gamma1, "d_beta1": d_beta1,
            "d_gamma2": d_gamma2, "d_beta2": d_beta2,
            "d_gamma3": d_gamma3, "d_beta3": d_beta3,
            "self_a": [dq1, dk1, dv1, dwq1, dwk1, dwv1, dwo1],
            "cross_a": [dq2, dk2, dv2, dwq2, dwk2, dwv2, dwo2],
        }