# Author:       Emma Gillespie
# Date:         2025-04-30
# Description:  The file responsible for tokenizing the data being used for the AI.

class Tokenizer:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}

    def build_vocab(self, texts):
        for text in texts:
            for word in text.lower().split():
                idx = len(self.word2idx)

                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text, max_len):
        tokens = [self.word2idx.get(w, self.word2idx["<unk>"]) for w in text.lower().split()]
        tokens = [self.word2idx["<sos>"]] + tokens + [self.word2idx["<eos>"]]

        return tokens[:max_len] + [self.word2idx["<pad>"]] * (max_len - len(tokens))
    
    def decode(self, token_ids):
        words = [self.idx2word.get(tid, "<unk>") for tid in token_ids]
        return " ".join(w for w in words if w not in ("<pad>", "<sos>", "<eos>"))
    
    def parse_datasets(self, data):
        dataset_data = []
        inputs, outputs = [], []

        for src, tgt in data:
            dataset_data.append(src)
            dataset_data.append(tgt)

            inputs.append(src)
            outputs.append(tgt)

        return dataset_data, inputs, outputs