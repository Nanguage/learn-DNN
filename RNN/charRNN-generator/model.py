import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
import numpy as np


class CharRNN(chainer.Chain):
    def __init__(self, n_vocab, n_units, n_hidden=1):
        super(CharRNN, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units)
            self.l1 = L.LSTM(n_units, n_units)
            l_hidden = Sequential( L.LSTM(n_units, n_units) )
            self.l2 = l_hidden.repeat(n_hidden)
            self.l3 = L.Linear(n_units, n_vocab)
        
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

    def reset_state(self):
        self.l1.reset_state()
        for s in self.l2:
            for lstm in s:
                lstm.reset_state()

    def forward(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0))
        h2 = self.l2(F.dropout(h1))
        y = self.l3(F.dropout(h2))
        return y


if __name__ == "__main__":
    batch_size = 80
    a = np.arange(batch_size).astype(np.int)
    rnn = CharRNN(100, 10, 10)
    rnn(a)