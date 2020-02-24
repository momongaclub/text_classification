import collections
import numpy as np
import pandas as pd
import torchtext
import torch

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
        self.text = torchtext.data.Field()
        self.label = torchtext.data.Field()
        self.vocab = torchtext.data.Field()
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
        self.text.build_vocab(self.train_ds.Text, self.val_ds.Text, self.test_ds.Text,
                              vectors=torchtext.vocab.GloVe(name='6B', dim=300))
        print(self.text.vocab.vectors.size())

    def make_iter(self):
        self.train_iter, self.val_iter, self.test_iter = torchtext.data.BucketIterator.splits( 
            (self.train_ds, self.val_ds, self.test_ds), batch_sizes=(TRAIN_BATCH_SIZE, 
                                                                     VAL_BATCH_SIZE,
                                                                     TEST_BATCH_SIZE),
                                                                     )
                                                                     #device=-1)

    def make_dataloader(self):
        self.data_loader = torch.utils.data.DataLoader(self.train_ds)

    
def main():
    data_set = Data()
    data_set.make_dataset()
    data_set.make_vocab()
    data_set.make_iter()
    data_set.make_dataloader()
    print(data_set.train_ds[0].Text)
    print(data_set.train_ds[0].Label)
    print(data_set.data_loader)


    v_size = data_set.text.vocab.vectors.size()[0]
    emb_dim = data_set.text.vocab.vectors.size()[1]
    h_dim = 2
    model = Model.RNNmodel(emb_dim, h_dim, v_size, data_set.text.vocab.vectors)

if __name__ == '__main__':
    main()
