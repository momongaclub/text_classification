import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import Data

class RNNmodel(nn.Module):

    #def __init__(self, emb_dim, h_dim, v_size, gpu=True, batch_first=True):
    def __init__(self, emb_dim, h_dim, v_size, vocab_vectors, n_in, gpu=True):
        super(RNNmodel, self).__init__()
        self.gpu = gpu
        self.h_dim = h_dim
        self.embed = nn.Embedding(v_size, emb_dim)
        self.embed.weight.data.copy_(vocab_vectors)
        self.lstm = nn.LSTM(emb_dim, h_dim)
        self.fc1 = nn.Linear(n_in, n_out =1)

    def forward(self, x_in):
        y_out = self.fc1(x_in).squeeze()
        y_out = F.sigmoid(y_out)
        return y_out

def main():
    emb_dim = 3
    h_dim = 3
    v_size = 2
    vocab_vectors = 2
    model = RNNmodel(emb_dim, h_dim, v_size, vocab_vectors)


if __name__ == '__main__':
    main()
