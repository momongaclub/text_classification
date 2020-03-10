import sys
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import spacy
import matplotlib.pyplot as plt

# from Class import Model

PATH = '.data/ag_news_csv/'
# PATH = './Class/.data/ag_news_csv/'
TRAIN = 'train_sample.csv'
TRAIN = 'train.csv'
TEST = 'test.csv'
VALIDATION = 'test.csv'
FORMAT = 'csv'
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256


def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en.tokenizer(text)]

class Data():

    def __init__(self):
        # batch_first は [batch, x, x]のように一番最初にbatchの次元を持ってくる
        # self.text = torchtext.data.Field(sequential=True, batch_first=True, tokenize=tokenizer)
        self.text = torchtext.data.Field(sequential=True, batch_first=True, lower=True)
        self.head = torchtext.data.Field(sequential=True, batch_first=True, lower=True)
        self.label = torchtext.data.Field(batch_first=True)
        self.train_ds = 0
        self.val_ds = 0
        self.test_ds = 0
        self.train_iter = 0
        self.val_iter = 0
        self.test_iter = 0

    def make_dataset(self):
        self.train_ds, self.val_ds, self.test_ds = \
            torchtext.data.TabularDataset.splits(
                path=PATH, train=TRAIN, test=TEST,
                validation=VALIDATION, format=FORMAT,
                fields=[('Label', self.label), ('Head', self.head), ('Text', self.text)])

    def make_vocab(self):
        # 3種類の辞書を作成,n vectorsを指定すると事前学習したベクトルを読み込める
        self.text.build_vocab(self.train_ds.Text, self.val_ds.Text,
                              self.test_ds.Text, vectors=
                              torchtext.vocab.GloVe(name='6B', dim=300))
        self.head.build_vocab(self.train_ds.Head, self.val_ds.Head,
                              self.test_ds.Head, vectors=
                              torchtext.vocab.GloVe(name='6B', dim=300))
        self.label.build_vocab(self.train_ds.Label,
                               self.val_ds.Label, self.test_ds.Label)

    def make_iter(self):
        self.train_iter, self.val_iter, self.test_iter = \
            torchtext.data.Iterator.splits(
                (self.train_ds, self.val_ds, self.test_ds), batch_sizes=(
                    TRAIN_BATCH_SIZE,
                    VAL_BATCH_SIZE,
                    TEST_BATCH_SIZE), repeat=False
            )


class simplernn(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, vocab_vectors, gpu=False, batch_first=True):
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
        #y = torch.tanh(y)
        #y = self.softmax(y)
        #print(y[0])
        #print("hidden", hidden)
        return y

def main():
    data_set = Data()
    data_set.make_dataset()
    data_set.make_vocab()
    data_set.make_iter()

    vocab_size = data_set.text.vocab.vectors.size()[0]
    embedd_dim = data_set.text.vocab.vectors.size()[1]
    hidden_dim = 200
    vocab_vectors = data_set.text.vocab.vectors

    rnn = simplernn(embedd_dim, hidden_dim, vocab_size, vocab_vectors)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01, momentum=0.9)

    losses = []
    batch_sizes = []

    for epoch in range(100):
        data_len = len(data_set.train_iter)
        batch_len = 0
        for batch in iter(data_set.train_iter):
            batch_len = batch_len + 1
            target = batch.Label
            # torch.eye(クラス数)[対象tensor]でonehotへ
            target = torch.eye(6, dtype=torch.long)[target] # デフォルトfloatになるのでlong指定
            target = target.squeeze() #次元変換
            optimizer.zero_grad()
            output = rnn.forward(batch.Text) 
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            print('epoch:', epoch, 'batch_len', batch_len, '/', data_len, 'loss:', loss.item())
            batch_sizes.append(batch_len)
            losses.append(loss)
            #graph.set_data(batch_len, loss.item())
            plt.plot(batch_sizes, losses)
            plt.draw()
            plt.pause(0.1)
            plt.cla()


if __name__ == '__main__':
    main()
