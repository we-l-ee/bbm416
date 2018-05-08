from image_loader import *
from torch.autograd import Variable
from utils import *

import abc
import math

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torch


# Model is main class which includes training, model loading, model saving, testing etc.
# Train and test methods writes numpy data to outputs to be able to plot afterwards if it is wished.
#
class Model:
    def __init__(self, model_path, output_path, name, loss='mse', label_encoder=None, cuda=True, data_loader=None):
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

        if data_loader is None:
            data_loader = image_loader
        self.image_loader = data_loader

        if label_encoder is None:
            label_encoder = LabelEncoder()
        self.labelEncoder = label_encoder

        self.info = []
        self.is_freeze = False
        self.clip = None

        self.train_set = None
        self.train_loader = None
        self._train_batch_size = None
        self._train_set_len = None

        self.test_set = None
        self.test_loader = None

        self._test_set_len = None
        self._test_batch_size = None

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
                    if self.is_freeze:
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
        self.train_set = Dataset.load(train_path)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=batch_size, shuffle=True,
                                                        num_workers=0)

        self._train_batch_size = batch_size
        self._train_set_len = len(self.train_set)

    def train(self, epoch=5, lr=0.001, momentum=0.9, write=True):
        if epoch == 0:
            return

        print("Training starting...")
        self.adjust_last_layer()

        self.eta.set_epoch(epoch)
        self.eta.set_totiter(math.ceil(self._train_set_len / self._train_batch_size))

        self.model.train()

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=momentum)

        losses = []
        errs = []
        for e in range(1, epoch + 1):
            e_loss = []
            tot_loss = 0.0
            err = np.array([0, 0])
            for i, data in enumerate(self.train_loader):
                tot_loss, _err = self.__iter_train(e, i + 1, data, tot_loss, optimizer, e_loss)
                err += _err
            err = 100 * err / self._train_set_len
            errs.append(err)
            print("Epoch", e, 'completed! Top 1 and 5 Error Percentage is [%.2f %.2f]' % (err[0], err[1]))
            losses.append(e_loss)
        print('Training Completed.')

        if write:
            curr_dir = os.path.join(self.output_path, self.name)
            save = {'type': 'train', 'config': [self._train_batch_size, lr, momentum], 'loss': losses, 'error': errs}
            if os.path.isfile(curr_dir + '.npy'):
                prev = np.load(curr_dir + ".npy")
                prev = prev.tolist()
            else:
                prev = []
            prev.append(save)
            np.save(curr_dir, prev)
        self.info.append(['train', [epoch, self._train_batch_size, lr, momentum, err[0], err[1]]])

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

        loss = loss.cpu()
        loss = loss.item()

        e_loss.append(loss)
        self.eta.end()
        eta = self.eta.eta()
        print("Epoch %d [%d/%d (%.2f%%)]" % (
            epoch, min(iter * self._train_batch_size, self._train_set_len), self._train_set_len,
            100 * iter / self.eta.totiter), "| Loss [%.6f]" % (e_loss[-1]),
              "ETA: %.2f min" % eta)
        return tot_loss, err

    def update_test_dataset(self, test_path='test.txt', batch_size=16):
        print("Test dataset is loading...")

        self.test_set = DatasetText(test_path, self.labelEncoder, loader=self.image_loader)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=False, num_workers=0)

        self._test_set_len = len(self.test_set)
        self._test_batch_size = batch_size

    def test(self, write=True):
        self.eta.set_epoch(0)
        self.eta.set_totiter(math.ceil(self._test_set_len / self._test_batch_size))

        self.model.eval()

        err = np.array([0, 0])
        print("Test starting...")
        for i, data in enumerate(self.test_loader):
            err += self.__iter_test(i, data)
        err = 100 * err / self._test_set_len
        self.info.append(['test', [err[0], err[1]]])

        if write:
            curr = os.path.join(self.output_path, self.name)
            save = {'type': 'test', 'error': err}
            if os.path.isfile(curr + '.npy'):
                prev = np.load(curr + ".npy")
                prev = prev.tolist()
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
        if self.cuda:
            labels = labels.cuda()

        err = self.cal_top_errors(outputs, labels)

        self.eta.end()
        eta = self.eta.eta()
        curr_batch = min(iter * self._test_batch_size, self._test_set_len)
        bath_size = curr_batch - (iter - 1) * self._test_batch_size
        err_per = 100 * err / bath_size
        print("Testing... [%d/%d (%.2f%%)]" % (
            curr_batch, self._test_set_len, 100 * iter / self.eta.totiter),
              "| Error percentages [%.2f %.2f]" % (err_per[0], err_per[1]), "ETA: %.2f min" % (eta))

        return err

    @abc.abstractmethod
    def adjust_last_layer(self):
        raise NotImplementedError("For output layer, it must be implemented.")

    def freeze(self, ind):
        self.is_freeze = True
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

    def __init__(self, model_path, output_path, name, label_encoder=None, cuda=True, model=None, loss='mse',
                 batch_norm=False):

        super(VGGModel, self).__init__(model_path=model_path, output_path=output_path, name=name, label_encoder=label_encoder,
                         cuda=cuda, loss=loss, data_loader=None)

        if model is None:
            if batch_norm:
                model = models.vgg16_bn(pretrained=True)
            else:
                model = models.vgg16(pretrained=True)

        if cuda:
            self.model = model.cuda()
        else:
            self.model = model

    @staticmethod
    def load(loc, model_path, output_path, name, cuda, loss, args):
        if len(args) > 0:
            __type = args[0]

        le = LabelEncoder.load(loc + ".npy")
        model = models.vgg16(num_classes=le.len())
        pt = torch.load(loc + ".pt")
        model.load_state_dict(pt)
        return VGGModel(model_path, output_path, name, label_encoder=le, cuda=cuda, model=model, loss=loss)

    def adjust_last_layer(self):

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

        if self.cuda:
            new = new.cuda()

        self.model.classifier = new

        for param1, param2 in zip(self.model.classifier.parameters(), old.parameters()):
            if len(param1) == len(param2):
                param1.data = param2.data
