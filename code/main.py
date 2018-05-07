import os
import argparse
import gc
import time
import math
import sys

from os import path

import abc

import torch.optim as optim
import torchvision.models as models
import torch
import torchvision.transforms as transforms

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

import matplotlib.pyplot as plt

import numpy as np

from torchvision.models.vgg import VGG

test = []

# Volatile variables to be used in inference only. They get rid off other variables that is used in back propagation.
#
gc.enable()


class ETA():
    def __init__(self):
        self.est = 0
        self.epoch = 0
        self.ep = 0
        self.totiter = 0
        self.iter = 0
        self.s = 0;
        self.e = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_totiter(self, totiter):
        self.totiter = totiter

    def update(self, ep, iter):
        self.ep = ep;
        self.iter = iter;

    def start(self):
        self.s = time.time()

    def end(self):
        self.e = time.time()

    # curr metric is second returns min estimate.
    def eta(self):
        curr = self.e - self.s
        self.est = 0.125 * curr + 0.875 * self.est
        return (self.est * ((self.epoch - self.ep) * self.totiter + (self.totiter - self.iter))) / 60


class LabelEncoder():
    def __init__(self, encode=None, max=0):
        if encode is None:
            encode = dict()
        self.encode = encode
        self.max = max

    @staticmethod
    def load(path):
        en, m = np.load(path)
        return LabelEncoder(en, m)

    def save(self, path):
        np.save(path, [self.encode, self.max])

    def changeLabelCode(self, label, code):
        self.encode[label] = code

    def encodeLabel(self, label):
        if label not in self.encode:
            self.encode[label] = self.max
            self.max += 1

    def decodeLabel(self, label):
        return self.encode[label]

    def len(self):
        return self.max


TEST = True


# Some images has alpha channel to. All converted to RGB.
def image_loader(image_path):
    image = Image.open(image_path).convert("RGB")
    # image = np.array(image); image = image/ np.linalg.norm(image); image = Image.fromarray(Image.)
    imsize = (224, 224)
    trans = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])

    return trans(image)


def image_loader_cuda(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")

    imsize = (224, 224)
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    image = loader(image)

    return image.cuda()


class DatasetText(Dataset):
    def __init__(self, path, labelEncoder, root='Images', ImageLoader=image_loader):
        _in = open(path, 'r')
        self.data = []
        self.label = []

        self.trained = []

        for line in _in.readlines():
            self.data.append(
                ImageLoader(os.path.join(root, line.strip()))
            )
            label = os.path.split(os.path.split(line)[0])[1]
            labelEncoder.encodeLabel(label)
            self.label.append(
                labelEncoder.decodeLabel(label)
            )
        self.len = len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.len


# Model is main class which includes traning, model loading, model saving, testing etc.
# Train and test methods writes numpy data to outputs to be able to plot afterwards if it is wished.
#
class Model:
    def __init__(self, model_path, output_path, name, loss='mse', labelEncoder=None, cuda=True, ImageLoader=None):
        self.model_path = model_path
        self.output_path = output_path
        self.name = name

        self.eta = ETA()

        self.__set_loss(loss)

        self.cuda = cuda

        if cuda:
            self.variable = self.__variable_cuda
        else:
            self.variable = self.__variable

        if ImageLoader is None: ImageLoader = image_loader
        self.image_loader = ImageLoader


        if labelEncoder is None:
            labelEncoder = LabelEncoder()
        self.labelEncoder = labelEncoder

        self.info = []
        self.freezed = False;
        self.clip = None
        print("Model Initialized.")

    def __set_loss(self, loss):
        if loss == 'cross':
            self.func_target = self.__cross_output_target
            self.criterion = torch.nn.CrossEntropyLoss()
        elif loss == 'mse':
            self.func_target = self.__mse_output_target
            self.criterion = torch.nn.MSELoss()

    def __variable_cuda(self, inputs, labels):
        return Variable(inputs).cuda(), Variable(labels).cuda()

    def __variable(self, inputs, labels):
        return Variable(inputs), Variable(labels)

    def save(self):
        loc = os.path.join(self.model_path, self.name)
        torch.save(self.model.state_dict(), loc + ".pt")
        self.labelEncoder.save(loc + ".npy")

    def save_info(self):
        curr = os.path.join(self.model_path, self.name)
        is_new = not os.path.isfile(curr + ".info")

        with open(curr + '.info', 'ab') as finfo:
            if is_new:
                finfo.write(bytes(self.__class__.__name__, 'utf-8') + b'\n')

            for action, output in self.info:
                if action == 'train':
                    finfo.write(b"Trained:\n")
                    if self.freezed:
                        finfo.write(b"Freezed layers between (%d-%d):\n" % (self.clip[0], self.clip[1]))
                    finfo.write(b'Epoch: %d' % output[0] + b'\t|\t'
                                                           b'Batch size: %d' % output[1] + b'\t|\t'
                                                                                           b'Learning rate: %f' %
                                output[2] + b'\t|\t'
                                            b'Momentum: %f' % output[3] + b'\t|\t'
                                                                          b'Top 1 Error: %.2f' % output[4] + b'\t|\t'
                                                                                                             b'Top 5 Error: %.2f' %
                                output[5] + b'\n')
                elif action == "test":
                    finfo.write(b"Test:\n")
                    finfo.write(b'Error Top 1 %.2f%%' % output[0] + b'\t|\t'
                                                                    b'Error Top 5 %.2f%%' % output[1] + b'\n')
        self.info = []

    def update_train_dataset(self, train_path='train.txt', batch_size=16):
        print("Train dataset is loading...")
        self.trainset = DatasetText(train_path, self.labelEncoder, ImageLoader=self.image_loader)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True,
                                                       num_workers=0)

        self._trainbatch_size = batch_size
        self._trainsetlen = len(self.trainset)

    def train(self, epoch=5, lr=0.001, momentum=0.9, write=True):
        if epoch == 0:
            return

        print("Training starting...")
        self.adjust_lastlayer()

        self.eta.set_epoch(epoch)
        self.eta.set_totiter(math.ceil(self._trainsetlen / self._trainbatch_size))

        self.model.train()

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=momentum)


        losses = []
        errs = []
        for e in range(1, epoch + 1):
            e_loss = []
            tot_loss = 0.0
            err = np.array([0, 0])
            for i, data in enumerate(self.trainloader):
                tot_loss, _err = self.__iter_train(e, i + 1, data, tot_loss, optimizer, e_loss)
                err += _err
            err = 100 * err / self._trainsetlen
            errs.append(err)
            print("Epoch", e, 'completed! Top 1 and 5 Error Percentage is [%.2f %.2f]' % (err[0], err[1]))
            losses.append(e_loss)
        print('Training Completed.')

        if write:
            curr_dir = os.path.join(self.output_path, self.name)
            save = {'type': 'train', 'config': [self._trainbatch_size, lr, momentum], 'loss': losses, 'error': errs}
            if os.path.isfile(curr_dir + '.npy'):
                prev = np.load(curr_dir + ".npy"); prev = prev.tolist();
            else:
                prev = []
            prev.append(save)
            test.append(save)
            np.save(curr_dir, prev)
        self.info.append(['train', [epoch, self._trainbatch_size, lr, momentum, err[0], err[1]]])

    def __mse_output_target(self, outputs, targets):
        if self.cuda:
            targets = Variable(
                torch.from_numpy(np.eye(self.labelEncoder.len(), dtype=np.dtype(float))[targets]).float()).cuda()
        else:
            targets = Variable(
                torch.from_numpy(np.eye(self.labelEncoder.len(), dtype=np.dtype(float))[targets]).float())
        return torch.nn.functional.softmax(outputs, dim=1), targets

    def __cross_output_target(self, outputs, targets):
        return outputs, targets

    def __iter_train(self, epoch, iter, data, tot_loss, optimizer, e_loss):
        self.eta.update(epoch, iter)
        self.eta.start()
        inputs, labels = data

        inputs, targets = self.variable(inputs, labels)

        optimizer.zero_grad()

        outputs = self.model(inputs)

        err = self.cal_top_errors(outputs, targets)

        # To get best targets for criterion method. For feature use if there is more than one loss function option.
        outputs, targets = self.func_target(outputs, targets)

        loss = self.criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss = loss.cpu();
        loss = loss.item()

        e_loss.append(loss)
        self.eta.end()
        eta = self.eta.eta()
        print("Epoch %d [%d/%d (%.2f%%)]" % (
            epoch, min(iter * self._trainbatch_size, self._trainsetlen), self._trainsetlen,
            100 * iter / self.eta.totiter), "| Loss [%.6f]" % (e_loss[-1]),
              "ETA: %.2f min" % (eta))
        return tot_loss, err

    def update_test_dataset(self, test_path='test.txt', batch_size=16):
        print("Test dataset is loading...")

        self.testset = DatasetText(test_path, self.labelEncoder, ImageLoader=self.image_loader)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=0)

        self._testsetlen = len(self.testset)
        self._testbatch_size = batch_size

    def test(self, write=True):
        self.eta.set_epoch(0)
        self.eta.set_totiter(math.ceil(self._testsetlen / self._testbatch_size))

        self.model.eval()

        err = np.array([0, 0])
        print("Test starting...")
        for i, data in enumerate(self.testloader):
            err += self.__iter_test(i, data)
        err = 100 * err / self._testsetlen
        self.info.append(['test', [err[0], err[1]]])

        if write:
            curr = os.path.join(self.output_path, self.name)
            save = {'type': 'test', 'error': err}
            if os.path.isfile(curr + '.npy'):
                prev = np.load(curr + ".npy"); prev = prev.tolist();
            else:
                prev = []
            prev.append(save)
            np.save(curr, prev)

        print("Error percentages of test [%.2f %.2f]" % (err[0], err[1]))
        print("Test ended!")

    def __iter_test(self, iter, data):
        self.eta.update(0, iter)
        self.eta.start()
        inputs, labels = data

        inputs, targets = self.variable(inputs, labels)

        outputs = self.model(inputs)
        if self.cuda: labels = labels.cuda()

        err = self.cal_top_errors(outputs, labels)

        self.eta.end()
        eta = self.eta.eta()
        curr_batch = min(iter * self._testbatch_size, self._testsetlen)
        bath_size = curr_batch - (iter - 1) * self._testbatch_size
        err_per = 100 * err / bath_size
        print("Testing... [%d/%d (%.2f%%)]" % (
            curr_batch, self._testsetlen, 100 * iter / self.eta.totiter),
              "| Error percentages [%.2f %.2f]" % (err_per[0], err_per[1]), "ETA: %.2f min" % (eta))

        return err

    @abc.abstractmethod
    def adjust_lastlayer(self):
        raise NotImplementedError("For output layer, it must be implemented.")

    def freeze(self, ind):
        self.freezed = True;
        self.clip = ind
        param = list(self.model.parameters())
        if ind[1] > len(param):
            ind[1] = len(param)

        for i in range(ind[0], ind[1]):
            param[i].requires_grad = False
            print('Layer', i, param[i].size(), 'freezed.')

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
            print(param.size(), 'unfreezed.')

    def cal_top_errors(self, o, t):
        _, p = torch.topk(o, 5)
        top1 = (p[:, 0] != t).sum().cpu().item()
        top5 = 0
        for i, ti in enumerate(t):
            top5 += int(ti not in p[i])

        return np.array([top1, top5])


class VGGModel(Model):

    def __init__(self, model_path, output_path, name, labelEncoder=None, cuda=True, model= None, loss='mse', batch_norm = False):

        super().__init__(model_path=model_path, output_path=output_path, name=name, labelEncoder=labelEncoder,
                         cuda=cuda, loss=loss, ImageLoader=None)

        if model is None:
            if batch_norm : model = models.vgg16_bn(pretrained=True)
            else: model = models.vgg16(pretrained=True)

        if cuda:
            self.model = model.cuda()
        else:
            self.model = model

    @staticmethod
    def load(loc, model_path, output_path, name, cuda, loss):
        le = LabelEncoder.load(loc + ".npy")
        model = models.vgg16(num_classes=le.len())
        pt = torch.load(loc + ".pt")
        model.load_state_dict(pt)
        return VGGModel(model_path, output_path, name, labelEncoder=le, cuda=cuda, model=model, loss=loss)

    def adjust_lastlayer(self):

        old = self.model.classifier
        new = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, self.labelEncoder.len())
        )

        if self.cuda: new = new.cuda()

        self.model.classifier = new

        for param1, param2 in zip(self.model.classifier.parameters(), old.parameters()):
            if len(param1) == len(param2):
                param1.data = param2.data


def load(model_path, output_path, lname, sname, cuda, loss):
    print("Model loading...")

    loc = os.path.join(model_path, lname)
    with open(loc + ".info", 'rb') as f:
        type = f.readline().strip()
    func = {b"VGGModel": VGGModel.load}
    return func[type](loc, model_path, output_path, sname, cuda, loss)


def init(type, model_path, output_path, name, cuda, loss, batch_norm):
    print("Model initilization...")
    func = {"vgg16": VGGModel}
    return func[type](model_path, output_path, name, cuda=cuda, loss = loss, batch_norm = batch_norm)


def plot_all(output_path, figure_folder, figname):
    outputs = np.load(output_path)
    train_outs = dict()
    test_outs = dict()
    for out in outputs:
        if out['type'] == 'train':
            try:
                train_outs['error1'].extend([err[0] for err in out['error']])
            except KeyError:
                train_outs['error1'] = [err[0] for err in out['error']]

            try:
                train_outs['error5'].extend([err[1] for err in out['error']])
            except KeyError:
                train_outs['error5'] = [err[1] for err in out['error']]

            try:
                train_outs['loss'].extend([l for loss in out['loss'] for l in loss])
            except KeyError:
                train_outs['loss'] = [l for loss in out['loss'] for l in loss]

            try:
                train_outs['config'].append(out['config'])
            except KeyError:
                train_outs['config'] = [out['config']]

        elif out['type'] == 'test':
            try:
                test_outs['error1'].append(out['error'][0])
            except KeyError:
                test_outs['error1'] = [out['error'][0]]

            try:
                test_outs['error5'].append(out['error'][1])
            except KeyError:
                test_outs['error5'] = [out['error'][1]]

    for k in train_outs:
        train_outs[k] = np.array(train_outs[k])
    for k in test_outs:
        test_outs[k] = np.array(test_outs[k])

    fig1 = plt.figure(1)
    plt.title("Train top 1 error percentages after each epoch")
    plt.plot(range(len(train_outs['error1'])), train_outs['error1'])
    plt.savefig(os.path.join(figure_folder, figname) + "-train-err1.png", format='png')

    fig2 = plt.figure(2)
    plt.title("Train top 5 error percentages after each epoch")
    plt.plot(range(len(train_outs['error5'])), train_outs['error5'])
    plt.savefig(os.path.join(figure_folder, figname) + "-train-err5.png", format='png')

    fig3 = plt.figure(3)
    plt.title("Test top 1 error percentages after each tranning")
    plt.plot(range(len(test_outs['error1'])), test_outs['error1'])
    plt.savefig(os.path.join(figure_folder, figname) + "-test-err1.png", format='png')

    fig4 = plt.figure(4)
    plt.title("Test top 5 error percentages after each tranning")
    plt.plot(range(len(test_outs['error5'])), test_outs['error5'])
    plt.savefig(os.path.join(figure_folder, figname) + "-test-err5.png", format='png')

    fig5 = plt.figure(5, figsize=(5, 3), dpi=500)
    plt.title("Loss of the training after each iteration")
    plt.scatter(y=train_outs['loss'], x=range(len(train_outs['loss'])), s=0.5)
    plt.savefig(os.path.join(figure_folder, figname) + "-train-loss.png", format='png')
    print("Plots are written under to './" + figure_folder + "/'!")


def main():
    parser = argparse.ArgumentParser(description='DNN Parser of BBM418 Assignment 3.')
    parser.add_argument("-ftrain", default='train.txt',
                        help="Text file which has the paths of all the train images."
                             " Default is 'train.txt'."
                        )
    parser.add_argument("-ftest", default='test.txt',
                        help="Text file which has the paths of all the test images."
                             " Default is 'test.txt'."
                        )
    parser.add_argument("-model_path", default='models',
                        help="Root folder of the stored models."
                             " Default is 'models'."
                        )
    parser.add_argument("-output_path", default='outputs',
                        help="Root folder of the any outputs of the models."
                             " Default is 'outputs'."
                        )
    parser.add_argument("-figure_path", default='figures',
                        help="Root folder of the any outputs of the plot figures."
                             " Default is 'figures'."
                        )

    parser.add_argument("-batch", type=int, default=16,
                        help="Batch size of the train data")
    parser.add_argument("-lr", type=float, default=0.01,
                        help="Learning rate of the training optimizer")
    parser.add_argument("-momentum", type=float, default=0.9,
                        help="Momentum value of the SGD optimizer")

    parser.add_argument("-save", action='store_true',
                        help="Save after the training. Model is named incrementally if there is already loaded model."
                             " Default is 'false', disabled."
                        )
    parser.add_argument("-load", default=None,
                        help="Load model before the training. This will be name for the loading."
                             " Default is 'None', disabled."
                        )
    parser.add_argument("-mname", default="model-1.0",
                        help="Model`s name that will be used in saving action."
                             " Default is 'model-1.0'.")

    parser.add_argument("-plot", action='store_true',
                        help="Plots the loss output by given -mname which is used with -output_path to determine which "
                             "loss output should be plotted.")

    parser.add_argument("-train", type=int, nargs='+', default=[0],
                        help="For every given integer it will run as that much epochs. After each run of epocs "
                             "if test is activated test will be applied to model. Default is single run of 50 epochs."
                             " Default is '0' which means disabled."
                        )

    parser.add_argument("-test", action='store_true',
                        help="Enables to test.")

    parser.add_argument("-freeze", action='store_true',
                        help="Enables freezing -clip param will change the freezed layers."
                             " Default is 'false', disabled.")
    parser.add_argument("-clip", type=int, nargs='+', default=[0, 26],
                        help="Freezes the indicated layers by given to int.[0,26] is the default between 0 and 25th "
                             "layers is freezed." " Default is '0 26'."
                        )

    parser.add_argument("-type", default='vgg16',
                        help="Initialized vgg16 model as default.")

    parser.add_argument("-cuda", action='store_true',
                        help="Use cuda.")

    parser.add_argument("-mkdir", action='store_true',
                        help="Activate it if giving directories are not made before hand.")

    parser.add_argument("-loss", default='mse',
                        help="Loss function. Default is mse.")

    parser.add_argument("-batch_norm", action='store_true',
                        help="Activates batch_normalization. It can only be used when new model is initialized")

    # print(sys.argv)
    args = parser.parse_args(sys.argv[1:])
    print(args)

    if args.mkdir:
        try:
            os.mkdir(args.model_path)
        except:
            pass
        try:
            os.mkdir(args.output_path)
        except:
            pass
        try:
            os.mkdir(args.figure_path)
        except:
            pass

    model = None
    if args.load is not None:
        model = load(args.model_path, args.output_path, args.load, args.mname, args.cuda, args.loss);
    elif (len(args.train) > 0 and args.train[0] > 0) or args.test:
        model = init(args.type, args.model_path, args.output_path, args.mname, args.cuda, args.loss, args.batch_norm);

    if args.freeze:
        model.freeze(args.clip)

    if len(args.train) > 0 and args.train[0] > 0:
        model.update_train_dataset(args.ftrain, args.batch)
        model.update_test_dataset(args.ftest, args.batch)

        for i, epoch in enumerate(args.train):
            print("Training of (", i + 1, "/", len(args.train), ") with epoch [", epoch, "] initializing...")
            model.train(epoch=epoch, lr=args.lr, momentum=args.momentum, write=True)
            print("Testing of (", i + 1, "/", len(args.train), ") initializing...")
            if args.test: model.test(write=True)

    elif args.test:
        model.update_test_dataset(args.ftest, args.batch); model.test(write=True)

    if model is not None:   model.save_info()
    if args.save: model.save()

    if args.plot:
        plot_all(path.join(args.output_path, args.mname) + '.npy', args.figure_path, args.mname)

    return model


# sys.argv.extend("-train 1 1 1 1 1 1 1 1 1 1 -load vgg-mse-10.32 -save -test -lr 0.01 -batch 32 -freeze -clip 0 26 -mname vgg-mse-10.32 -cuda".split())
# sys.argv.extend("-train 10 10 10 -save -test -lr 0.01 -batch 16 -mname vgg-mse-full-3.0 -cuda".split())
# sys.argv.extend("-test -load vgg-full-3.0 -cuda".split())
# sys.argv.extend("-plot -mname vgg-mse-5.0".split())
# sys.argv.extend("-plot -mname vgg-mse-full-2.0".split())

# sys.argv.extend("-train 1 1 -test -lr 0.01 -batch 32 -loss mse -freeze -clip 0 26 -cuda".split())

model = main()