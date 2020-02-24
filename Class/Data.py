import argparse
import collections
import numpy as np
import pandas as pd
import torchtext

PATH = './.data/ag_news_csv/'
TRAIN = 'train.csv'
TEST = 'test.csv'
VALIDATION = 'validation.csv'
FORMAT = 'csv'


class Data():

    def __init__(self):
        self.data = []
        self.Text = torchtext.data.Field()
        self.Label = torchtext.data.Field()
        self.train_ds = 0
        self.val_ds = 0
        self.test_ds = 0

    def make_dataset(self):
        self.train_ds, self.val_ds, self.test_ds = \
            torchtext.data.TabularDataset.splits(
                path=PATH, train=TRAIN, test=TEST,
                validation=VALIDATION, format=FORMAT,
                fields=[('Label', self.Label), ('Text', self.Text)])


def main():
    data_set = Data()
    data_set.make_dataset()
    print(data_set)
    print(vars(data_set.train_ds[0])['Text'])
    print(vars(data_set.train_ds[0])['Label'])

if __name__ == '__main__':
    main()
