import time
import numpy as np


class ETA(object):
    def __init__(self):
        self.est = 0
        self.epoch = 0
        self.ep = 0
        self.totiter = 0
        self.iter = 0
        self.s = 0
        self.e = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_totiter(self, totiter):
        self.totiter = totiter

    def update(self, ep, iter):
        self.ep = ep
        self.iter = iter

    def start(self):
        self.s = time.time()

    def end(self):
        self.e = time.time()

    # curr metric is second returns min estimate.
    def eta(self):
        curr = self.e - self.s
        self.est = 0.125 * curr + 0.875 * self.est
        return (self.est * ((self.epoch - self.ep) * self.totiter + (self.totiter - self.iter))) / 60


class LabelEncoder(object):
    def __init__(self, encode=None, max_=0):
        if encode is None:
            encode = dict()
        self.encode = encode
        self.max = max_

    @staticmethod
    def load(path):
        en, m = np.load(path)
        return LabelEncoder(en, m)

    def save(self, path):
        np.save(path, [self.encode, self.max])

    def change_label_code(self, label, code):
        self.encode[label] = code

    def encode_label(self, label):
        if label not in self.encode:
            self.encode[label] = self.max
            self.max += 1

    def decode_label(self, label):
        return self.encode[label]

    def len(self):
        return self.max
