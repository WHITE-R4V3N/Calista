# Author:       Emma Gillespie
# Date:         2025-04-30
# Description:  The file responsible for tokenizing the data being used for the AI.

class Tokenizer:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

    def build_vocab(self, texts):
        for text in texts:
            for word in text.lower().split():
                idx = len(self.word2idx)

                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text, max_len):
        tokens = [self.word2idx.get(w, self.word2idx["<UNK>"]) for w in text.lower().split()]
        tokens = [self.word2idx["<SOS>"]] + tokens + [self.word2idx["<EOS>"]]

        return tokens[:max_len] + [self.word2idx["<PAD>"]] * (max_len - len(tokens))
    
    def decode(self, token_ids):
        words = [self.idx2word.get(tid, "<UNK>") for tid in token_ids]
        return " ".join(w for w in words if w not in ("<PAD>", "<SOS>", "<EOS>"))