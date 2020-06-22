import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
import spacy
import matplotlib.pyplot as plt

PATH = './Class/.data/ag_news_csv/'
TRAIN = 'train_sample.csv'
TEST = 'test.csv'
VALIDATION = 'test.csv'
FORMAT = 'csv'
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 256
TEST_BATCH_SIZE = 16


def tokenizer(text):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en.tokenizer(text)]


class Data():

    def __init__(self):
        # batch_first は [batch, x, x]のように一番最初にbatchの次元を持ってくる
        self.text = torchtext.data.Field(
            sequential=True, batch_first=True, lower=True)
        self.head = torchtext.data.Field(
            sequential=True, batch_first=True, lower=True)
        self.label = torchtext.data.LabelField(batch_first=True, sequential=False)
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
        # 3種類の辞書を作成,vectorsを指定すると事前学習した分散表現を読み込める
        self.text.build_vocab(self.train_ds.Text, self.val_ds.Text,
                              self.test_ds.Text, vectors=torchtext.vocab.GloVe(name='6B', dim=300))
        self.head.build_vocab(self.train_ds.Head, self.val_ds.Head,
                              self.test_ds.Head, vectors=torchtext.vocab.GloVe(name='6B', dim=300))
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


def main():
    return 0


if __name__ == '__main__':
    main()
