import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('embeddings', type=str, help='include four keys.')
    args = parser.parse_args()
    return args

class simplernn(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, vocab_vectors, gpu=False, batch_first=True):
        """
        embedding_dim is dimention of vocab
        """
        # embedding_dimは分散表現の次元数,
        super(simplernn, self).__init__()
        self.gpu = gpu
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.embed.weight.data.copy_(vocab_vectors)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=batch_first)
        output_dim = 6
        self.linear1 = nn.Linear(hidden_dim, output_dim)
        #self.softmax = nn.LogSoftmax(dim=1) # TODO
        self.softmax = nn.LogSoftmax() # TODO

    def forward(self, sentence):
        # lstmの最初の入力に過去の隠れ層はないのでゼロベクトルを代入する
        # self.hidden = self.init_hidden(sentence.size(0))
        embed = self.embed(sentence)
        y, hidden = self.lstm(embed)
        y = torch.tanh(y)
        y = self.linear1(y)
        return y

def main():
    return 0


if __name__ == '__main__':
    main()
