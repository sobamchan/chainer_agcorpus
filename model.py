import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import chainer

from sobamchan.sobamchan_chainer import Model
from sobamchan.sobamchan_chainer_link import PreTrainedEmbedId

class MLP(Model):

    def __init__(self, class_n, vocab_n, d, vocab, fpath):
        super(MLP, self).__init__(
            embed=PreTrainedEmbedId(vocab_n, d, vocab, fpath, False),
            fc1=L.Linear(None, 100),
            fc2=L.Linear(100, 100),
            fc3=L.Linear(100, class_n),
        )

    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)

    def fwd(self, x, train):
        h = self.embed(x)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = self.fc3(h)
        return h

class CNN(Model):

    def __init__(self, class_n, vocab_n, d, vocab, fpath):
        super(CNN, self).__init__(
            embed=PreTrainedEmbedId(vocab_n, d, vocab, fpath, False),
            conv1=L.Convolution2D(1, 16, (1, 3)),
            conv2=L.Convolution2D(16, 3, (1, 3)),
            fc=L.Linear(None, class_n)
        )

    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)

    def fwd(self, x, train):
        h = self.embed(x)
        batch, height, width = h.shape
        h = F.reshape(h, (batch, 1, height, width))
        h = self.conv1(h)
        h = F.max_pooling_2d(self.conv2(h), (1, 3))
        h = self.fc(h)
        return h
