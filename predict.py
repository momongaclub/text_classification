from Class import Data
from Class import Model

import sys
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import classification_report


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
    print(device)

    data_set = Data.Data()
    data_set.make_dataset()
    data_set.make_vocab()
    data_set.make_iter()

    vocab_size = data_set.text.vocab.vectors.size()[0]
    embedd_dim = data_set.text.vocab.vectors.size()[1]
    hidden_dim = 100
    vocab_vectors = data_set.text.vocab.vectors

    rnn = Model.simplernn(embedd_dim, hidden_dim, vocab_size, vocab_vectors)
    # パラメータの読み込み
    parameter = torch.load('./model_weight/model1.pt',
                           map_location=torch.device(device))
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
        input_ = batch.Text
        input_ = input_.to(device)
        batch_outputs = rnn.forward(input_)
        target = batch.Label
        # print(input_, target)
        # print(batch_outputs, target)
        # print(target)
        # ラベルをone-hotベクトルへ変換
        # target = torch.eye(6, dtype=torch.long)[target]
        # print(target)
        target = target.squeeze()  # 次元変換
        target = target.to(device)
        #print('batch_len', batch_len, '/', 'data_len', data_len)
        # print('predict', batch_outputs.argmax(), 'target', target)
        # リスト内包表記　二重
        for outputs in batch_outputs:
            for output in outputs:
                p = int(output.argmax())
                pred.append(p)
        # print(output[0])
        #pred += [int(output.argmax()) for output in outputs]
        # print('pred', pred)
        print('output_len', len(pred), 'target_len', len(Y))
        Y += [int(t) for t in target]
        # print('predict', pred, 'target', Y)
        batch_sizes.append(batch_len)
        batch_len = batch_len + 1

    print(classification_report(Y, pred))
    """
    for p, t in zip(pred, Y):
        sum_ += 1
        if p == t:
            cnt += 1
    print(sum_, cnt)
    """


if __name__ == '__main__':
    main()
