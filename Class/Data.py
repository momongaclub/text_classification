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

    def load_data(self, fname):
        for line in fname:
            line = line

    def make_dataflame(self):
        self.train_ds, self.val_ds, self.test_ds = \
            torchtext.data.TabularDataset.splits(
                path=PATH, train=TRAIN, test=TEST,
                validation=VALIDATION, format=FORMAT,
                fields=[('Label', self.Label), ('Text', self.Text)])


def main():
    data_flame = Data()
    data_flame.make_dataflame()
    return 0


if __name__ == '__main__':
    main()
