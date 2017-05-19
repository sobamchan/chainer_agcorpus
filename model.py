import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Variable
import chainer

from sobamchan.sobamchan_chainer import Model
from sobamchan.sobamchan_chainer_link import PreTrainedEmbedId
from sobamchan.sobamchan_vocabulary import Vocabulary

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
            conv1=L.Convolution2D(1, 16, (3, 1)),
            conv2=L.Convolution2D(16, 3, (3, 1)),
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
        h = self.conv2(h)
        h = F.max_pooling_2d(h, (1, 3))
        h = self.fc(h)
        return h


class ResCNN(Model):

    def __init__(self, class_n, vocab_n, d, vocab, fpath):
        super(ResCNN, self).__init__(
            embed=PreTrainedEmbedId(vocab_n, d, vocab, fpath, False),
            conv1=L.Convolution2D(1, 16, (7, 1)),
            conv2=L.Convolution2D(16, 1, (7, 1)),
            fc=L.Linear(None, class_n)
        )

    def __call__(self, x, t, train=True):
        x = self.fwd(x, train)
        return F.softmax_cross_entropy(x, t), F.accuracy(x, t)

    def fwd(self, x, train):
        h = self.embed(x)
        batch, height, width = h.shape
        h = F.reshape(h, (batch, 1, height, width))
        org = h
        h = self.conv1(h)
        h = self.conv2(h)

        # arrange shape
        batch, h_c, h_h, h_w = h.shape
        _, org_c, org_h, org_w = org.shape
        if org_h != h_h:
            pad = Variable(np.zeros((batch, h_c, org_h - h_h, h_w)).astype(np.float32), volatile=org.volatile)
            h = F.concat((h, pad), 2)

        h += org
        h = F.max_pooling_2d(h, (1, 3))
        h = self.fc(h)
        return h

def test_ResCNN():
    vocab = Vocabulary()
    fpath = '/Users/sochan/project/ML/NLP/datas/word2vec_text8.txt'
    words = ['dog', 'cat', 'cow', 'sheep', 'sobamchan']
    for word in words:
        vocab.new(word)
    rescnn = ResCNN(2, 5, 300, vocab, fpath)

    x = Variable(np.array([[1,2,2,2,3,2,1,2,3,2,1,2,3,2,], [2,3,3,4,1,2,3,4,2,2,3,1,2,3]]).astype(np.int32))
    t = Variable(np.array([0, 1]).astype(np.int32))
    e, a = rescnn(x, t)
