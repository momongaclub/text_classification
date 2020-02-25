import collections
import numpy as np
import pandas as pd
import torchtext
import torch

import torch.nn as nn

import Model

PATH = './.data/ag_news_csv/'
TRAIN = 'train.csv'
TEST = 'test.csv'
VALIDATION = 'validation.csv'
FORMAT = 'csv'
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256


class Data():

    def __init__(self):
        self.dataloader = 0
        self.text = torchtext.data.Field(sequential=True,
                                         lower=True, include_lengths=True)
        self.label = torchtext.data.Field()
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
                fields=[('Label', self.label), ('Text', self.text)])

    def make_vocab(self):
        self.text.build_vocab(self.train_ds.Text, self.val_ds.Text,
                              self.test_ds.Text, vectors = \
                              torchtext.vocab.GloVe(name = '6B', dim = 300))
        self.label.build_vocab(self.train_ds.Text)

    def make_iter(self):
        self.train_iter, self.val_iter, self.test_iter = torchtext.data.Iterator.splits( 
            (self.train_ds, self.val_ds, self.test_ds), batch_sizes=(
                                                                TRAIN_BATCH_SIZE, 
                                                                VAL_BATCH_SIZE,
                                                                TEST_BATCH_SIZE),
                                                                )


def main():
    data_set = Data()
    data_set.make_dataset()
    data_set.make_vocab()
    data_set.make_iter()

    v_size = data_set.text.vocab.vectors.size()[0]
    emb_dim = data_set.text.vocab.vectors.size()[1]
    h_dim = 10
    model = Model.RNNmodel(emb_dim, h_dim, v_size,
                           data_set.text.vocab.vectors, 16)

    epochs = 0
    for batch in iter(data_set.train_iter):
        x, y = batch.Text[0], batch.Label
        out_ = model.forward(x)
        print(epochs, out_)
        epochs+=1


if __name__ == '__main__':
    main()
