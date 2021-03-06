import time
import numpy as np
import json
from sklearn.metrics import fbeta_score, matthews_corrcoef


def best_threshold_separately(outputs, targets, verbose=False):
    threshold = np.arange(0.1, 0.9, 0.1)

    acc = []
    accuracies = []
    best_threshold = np.zeros(outputs.shape[1])
    for i in range(outputs.shape[1]):
        y_prob = np.array(outputs[:, i])
        for j in threshold:
            y_pred = [1 if prob >= j else 0 for prob in y_prob]
            acc.append(matthews_corrcoef(targets[:, i], y_pred))
        acc = np.array(acc)
        index = np.where(acc == acc.max())
        accuracies.append(acc.max())
        best_threshold[i] = threshold[index[0][0]]
        acc = []
    if verbose:
        print("Matthews Correlation Coefficient")
        print("Class wise accuracies")
        print(accuracies)
    return best_threshold


def get_prediction_each_label_threshold(outputs, best_threshold):
    return np.array([[1 if outputs[i, j] >= best_threshold[j] else 0
                    for j in range(outputs.shape[1])] for i in range(len(outputs))])


def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def find_f2score_threshold(p_valid, y_valid, try_all=False, verbose=False):
    best = 0
    best_score = -1
    totry = np.arange(0, 1, 0.005) if try_all is False else np.unique(p_valid)
    for t in totry:
        score = f2_score(y_valid, p_valid > t)
        if score > best_score:
            best_score = score
            best = t
    if verbose is True:
        print('Finding Threshold-> Best score: ', round(best_score, 5), ' @ threshold =', best)
    return best


def label_info(path='train.json'):
    with open(path) as f:
        annots = json.load(f)["annotations"]

    set_ids = set()
    for an in annots:
        ids = an["labelId"]
        ids = [int(id) for id in ids]
        set_ids.update(ids)

    labels = np.array(list(set_ids))
    max_label = np.max(labels)
    min_label = np.min(labels)

    # for label in range(min_label, max_label+1):
    #     if label not in labels:
    #         raise Exception("data is invalidate")

    return labels, int(min_label), int(max_label)


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

    def __call__(self):

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
