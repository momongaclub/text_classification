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
    loss_function = nn.CrossEntropyLoss() # softmaxを含む
    optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01, momentum=0.9)
    rnn.to(device)

    losses = []
    batch_sizes = []

    for epoch in range(100):
        data_len = len(data_set.train_iter)
        batch_len = 0
        for batch in iter(data_set.train_iter):
            batch_len = batch_len + 1
            input_ = batch.Text
            input_ = input_.to(device)
            target = batch.Label
            # torch.eye(クラス数)[対象tensor]でonehotへ
            target = torch.eye(6, dtype=torch.long)[target] # デフォルトfloatになるのでlong指定
            target = target.squeeze() #次元変換
            target = target.to(device)
            optimizer.zero_grad()
            output = rnn.forward(input_)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            print('epoch:', epoch, 'batch_len', batch_len, '/', data_len, 'loss:', loss.item())
            batch_sizes.append(batch_len)
            losses.append(loss)
            #plot_progress(batch_sizes, losses)
            #if batch_len % 100 == 0:
        torch.save(rnn.state_dict(), './model_weight/' 'model' + str(epoch) + '.pt')


if __name__ == '__main__':
    main()
