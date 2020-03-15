from Class import Data
from Class import Model

import sys
import argparse
import torch
import torch.nn as nn


def plot_progress(x_axis, y_axis, time=0.1):
    """
    x_axis and y_axis is list
    """
    plt.plot(x_axis, y_axis)
    plt.draw()
    plt.pause(time)
    plt.cla()

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(device)

    data_set = Data.Data()
    data_set.make_dataset()
    data_set.make_vocab()
    data_set.make_iter()

    vocab_size = data_set.text.vocab.vectors.size()[0]
    embedd_dim = data_set.text.vocab.vectors.size()[1]
    hidden_dim = 200
    vocab_vectors = data_set.text.vocab.vectors

    rnn = Model.simplernn(embedd_dim, hidden_dim, vocab_size, vocab_vectors)
    # パラメータの読み込み
    parameter = torch.load('./Class/model_weight/model0.pt', map_location=torch.device(device))
    rnn.load_state_dict(parameter)

    # 評価モードに変更
    rnn = rnn.eval()
    rnn.to(device)

    losses = []
    batch_sizes = []

    pred = []
    Y = []

    cnt = 0
    sum_ = 0

    batch_len = 0
    data_len = len(data_set.train_iter)

    for batch in iter(data_set.train_iter):
        with torch.no_grad():
            batch_len = batch_len + 1
            input_ = batch.Text
            input_ = input_.to(device)
            batch_outputs = rnn.forward(input_)
            target = batch.Label
            target = torch.eye(6, dtype=torch.long)[target] # デフォルトfloatになるのでlong指定
            target = target.squeeze() #次元変換
            target = target.to(device)
            print('batch_len', batch_len,'/','data_len', data_len)
            #print('predict', output.argmax(), 'target', target)
            pred += [int(outputs.argmax()) for outputs in batch_outputs]
            Y += [int(t.argmax()) for t in target]
            #print('predict', pred, 'target', Y)
            batch_sizes.append(batch_len)
    for p, t in zip(pred, Y):
        sum_ += 1
        if p == t:
            cnt += 1
    print(sum_, cnt)


if __name__ == '__main__':
    main()
