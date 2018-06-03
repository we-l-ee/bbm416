from image_loader import *
from torch.autograd import Variable
from utils import *
import math

import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import torch
from torch.nn.functional import sigmoid, softmax
from sklearn.metrics import f1_score


# Train and test methods writes numpy data to outputs to be able to plot afterwards if it is wished.

class ModelOperator:
    def __init__(self, model, model_path, output_path, name, loss='mse', cuda=True):

        self.model = model
        if cuda:
            self.model.to_cuda()

        self.model_path = model_path

        self.name = name
        self.loc = os.path.join(self.model_path, self.name)

        self.output_path = output_path
        self.eta = ETA()
        self.__set_loss(loss)
        self.cuda = cuda

        if cuda:
            self.variable = self.__variable_cuda
        else:
            self.variable = self.__variable

        self.info = []
        self.is_freeze = False
        self.clip = None

        self._train_batch_size = None
        self._train_set_len = None

        self._test_set_len = None
        self._test_batch_size = None

        self._val_set_len = None
        self._val_batch_size = None

        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.scores = {'threshold_5': list(),
                       'threshold_6': list(),
                       'threshold_7': list(),
                       'threshold_8': list(),
                       'threshold_9': list(),
                       'best_threshold': list(),
                       'top_labels': list()}

        print("Model Initialized.")

    def __default_loss_layer_func(self, preds):
        return preds

    def __set_loss(self, loss):
        if loss == 'cross':
            self.func_target = self.__cross_output_target
            self.criterion = torch.nn.CrossEntropyLoss()
            self.loss_layer_func = self.__default_loss_layer_func
        elif loss == 'mse':
            self.func_target = self.__mse_output_target
            self.criterion = torch.nn.MSELoss()
            self.loss_layer_func = softmax
        elif loss == 'mlsm':
            self.func_target = self.__mlsm_output_target
            self.criterion = torch.nn.MultiLabelSoftMarginLoss()
            self.loss_layer_func = sigmoid
        elif loss == 'bce':
            self.func_target = self.__mlsm_output_target
            self.criterion = torch.nn.BCEWithLogitsLoss()
            self.loss_layer_func = sigmoid

    def __variable_cuda(self, inputs, labels, requires_grad=True):
        return Variable(inputs, requires_grad=requires_grad).cuda(), \
               Variable(labels, requires_grad=requires_grad).cuda()

    def __variable(self, inputs, labels, requires_grad=True):
        return Variable(inputs, requires_grad=requires_grad), Variable(labels, requires_grad=requires_grad)

    def save(self):
        self.model.save(self.loc)

    def save_info(self):
        curr = os.path.join(self.model_path, self.name)
        is_new = not os.path.isfile(curr + ".info")

        with open(curr + '.info', 'ab') as finfo:
            if is_new:
                finfo.write(self.model.identifier() + b'\n')

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
                                                                          b'Last Loss: %.2f' % output[4] + b'\n')
                elif action == "test":
                    finfo.write(b"Test:\n")
                    finfo.write(b'Error Top 1 %.2f%%' % output[0] + b'\t|\t'
                                                                    b'Error Top 5 %.2f%%' % output[1] + b'\n')
        self.info = []

    def update_train_dataset(self, train_path='train', batch_size=16, val_ratio=None, dtype='default', subsample=0):
        print("Train dataset is loading...")

        if dtype == 'subrandom':
            train = SubRandomDataSetFolder(subsample)

        elif dtype == 'default':
            train = DataSetFolder()

        elif dtype == 'lazy':
            train = LazyLoaderDataSet(subsample)
        else:
            raise Exception('Wrong type of dataset loading!!')

        validation = None
        if val_ratio is not None:
            if dtype == 'lazy':
                validation = LazyLoaderDataSet()
            else:
                validation = DataSetFolder()
        self.model.create_datasets(train=train, validation=validation)

        val_dirs = self.model.train_data_set.load(train_path, mode='train', val_ratio=val_ratio)

        self.model.adjust_last_layer(cuda=self.cuda)
        self.train_loader = torch.utils.data.DataLoader(self.model.train_data_set, batch_size=batch_size, shuffle=True,
                                                        num_workers=4)

        self._train_batch_size = batch_size
        self._train_set_len = len(self.model.train_data_set)
        if val_dirs is not None:
            self.model.val_data_set.update_label_info(self.model.train_data_set.num_classes, self.model.train_data_set.
                                                      min_class, self.model.train_data_set.max_class)

            self.model.val_data_set.load(train_path, mode='val', data_dirs=val_dirs)
            self.val_loader = DataLoader(self.model.val_data_set, batch_size=batch_size, shuffle=False, num_workers=4)

            self._val_set_len = len(self.model.val_data_set)
            self._val_batch_size = batch_size

    def __get_optimizer(self, optim_str, lr, momentum):
        if optim_str == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr, momentum=momentum)
        elif optim_str == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        elif optim_str == 'adagrad':
            optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        elif optim_str == 'adadelta':
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        elif optim_str == 'adamax':
            optimizer = optim.Adamax(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        elif optim_str == 'asgd':
            optimizer = optim.ASGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        else:
            raise Exception('Wrong optimizer')
        return optimizer

    def train(self, epoch=5, lr=0.001, momentum=0.9, write=True, optimizer='sgd'):
        if epoch == 0:
            return

        print("Training starting...")

        self.eta.set_epoch(epoch)
        self.eta.set_totiter(math.ceil(self._train_set_len / self._train_batch_size))

        self.model.train()
        optimizer = self.__get_optimizer(optimizer, lr, momentum)
        losses = []
        for e in range(1, epoch + 1):
            e_loss = []
            for i, data in enumerate(self.train_loader):
                self.__iter_train(e, i + 1, data, optimizer, e_loss)
            print("Epoch", e, 'completed!')
            losses.append(e_loss)
        print('Training Completed.')

        if write:
            curr_dir = os.path.join(self.output_path, self.name)
            save = {'type': 'train', 'config': [self._train_batch_size, lr, momentum], 'loss': losses,
                    'scores': self.scores}
            if os.path.isfile(curr_dir + '.npy'):
                prev = np.load(curr_dir + ".npy")
                prev = prev.tolist()
            else:
                prev = []
            prev.append(save)
            np.save(curr_dir, prev)
        self.info.append(['train', [epoch, self._train_batch_size, lr, momentum, losses[0][0]]])

    def __mse_output_target(self, outputs, targets):
        return softmax(outputs, dim=1), targets

    def __cross_output_target(self, outputs, targets):
        return outputs, targets

    def __mlsm_output_target(self, outputs, targets):
        return outputs, targets

    def __set_score_threshold(self, outputs, labels, threshold, key):
        if threshold == 'top':
            predicts = self.__encode_top_labels(labels, outputs)
            h_score = hamming_score(labels, predicts)
            f1_ = f1_score(labels, predicts, average='samples')
            print("hs-f1[%.2f %.2f] @ top labels" % (h_score * 100, f1_ * 100))
            return
        predictions = self.predict_with_loss_layer(outputs, threshold, return_binary=True)
        h_score = hamming_score(labels, predictions)
        f1_ = f1_score(labels, predictions, average='samples')
        self.scores[key].append([h_score, f1_])
        print("hs-f1[%.2f %.2f] @ t:%.2f" %
              (h_score * 100, f1_ * 100, threshold), end='//')

    def __encode_top_labels(self, labels, outputs_np):
        predicts = []
        for i, lab in enumerate(labels):
            target_indices = np.argwhere(lab == 1)
            predict_indices = np.argpartition(outputs_np, -len(target_indices))[-len(target_indices):]
            preds = np.zeros_like(lab)
            for ind in predict_indices:
                preds[ind] = 1
            predicts.append(preds)
        return np.array(predicts)

    def __set_scores(self, outputs, labels):
        labels = labels.detach().cpu().numpy()
        outputs_np = outputs.detach().cpu().numpy()

        best_threshold = find_f2score_threshold(outputs_np, labels, verbose=True)

        self.__set_score_threshold(outputs, labels, best_threshold, 'best_threshold')
        self.__set_score_threshold(outputs, labels, 0.5, 'threshold_5')
        self.__set_score_threshold(outputs, labels, 0.6, 'threshold_6')
        self.__set_score_threshold(outputs, labels, 0.7, 'threshold_7')
        self.__set_score_threshold(outputs, labels, 0.8, 'threshold_8')
        self.__set_score_threshold(outputs, labels, 0.9, 'threshold_9')
        self.__set_score_threshold(outputs_np, labels, 'top', 'top_labels')

    def __iter_train(self, epoch, _iter, data, optimizer, e_loss):
        self.eta.update(epoch, _iter)
        self.eta.start()
        inputs, labels = data

        inputs, targets = self.variable(inputs, labels)
        optimizer.zero_grad()
        outputs = self.model(inputs)

        outputs, targets = self.func_target(outputs, targets)
        loss = self.criterion(outputs, targets.detach().float())
        loss.backward()

        optimizer.step()

        loss = loss.cpu()
        loss = loss.item()

        e_loss.append(loss)
        self.__set_scores(outputs, targets)

        self.eta.end()
        eta = self.eta.eta()
        print("Epoch %d [%d/%d (%.2f%%)]" % (
            epoch, min(_iter * self._train_batch_size, self._train_set_len), self._train_set_len,
            100 * _iter / self.eta.totiter), "| Loss [%.6f]" % (e_loss[-1]),
              "ETA: %.2f min" % eta)

    def update_test_dataset(self, test_path='test', batch_size=16, dtype='default', subsample=0):
        print("Test dataset is loading...")

        if dtype == 'subrandom':
            self.model.create_datasets(test=SubRandomDataSetFolder(subsample))
        elif dtype == 'default':
            self.model.create_datasets(test=DataSetFolder())
        elif dtype == 'lazy':
            self.model.create_datasets(test=LazyLoaderDataSet(subsample))

        self.model.test_data_set.load(test_path, mode="test")

        self.test_loader = DataLoader(self.model.test_data_set, batch_size=batch_size, shuffle=False, num_workers=4)

        self._test_set_len = len(self.model.test_data_set)
        self._test_batch_size = batch_size

    def predict_with_loss_layer(self, predictions, threshold=0.5, return_binary=False):
        predictions = self.loss_layer_func(predictions).detach().cpu().numpy()
        indices = np.argwhere(predictions > threshold)
        return self.__encode_label_indices(predictions, indices, return_binary)

    def __encode_label_indices(self, labels, indices, return_binary):
        preds = len(labels) * [None]
        for ind in indices:
            index, label = ind[0], ind[1]
            if isinstance(preds[index], list):
                preds[index].append(label)
            else:
                preds[index] = [label]

        if return_binary:
            for i, pred in enumerate(preds):
                temp_pred = np.zeros(labels.shape[1], dtype=np.int)
                if pred is None:
                    preds[i] = temp_pred
                    continue
                for p in pred:
                    temp_pred[p] = 1
                preds[i] = temp_pred
        return np.array(preds)

    def test(self, write=True):
        self.eta.set_epoch(0)
        self.eta.set_totiter(math.ceil(self._test_set_len / self._test_batch_size))

        self.model.eval()

        predictions = list()
        ids = list()
        print("Test starting...")
        for i, data in enumerate(self.test_loader):
            outputs = self.__iter_test(i, data)
            predicts = self.predict_with_loss_layer(outputs, 0.5)
            predictions.extend(predicts)
            ids.append(data[1])

        if write:
            curr = os.path.join(self.output_path, self.name)
            save = {'type': 'test', 'prediction': predictions}
            if os.path.isfile(curr + '.npy'):
                prev = np.load(curr + ".npy")
                prev = prev.tolist()
            else:
                prev = []
            prev.append(save)
            np.save(curr, prev)

        print("Test ended!")
        return predictions, ids

    def __iter_test(self, _iter, data):
        self.eta.update(0, _iter)
        self.eta.start()
        inputs, _ = data

        inputs = Variable(inputs, requires_grad=False)
        outputs = self.model(inputs)

        self.eta.end()
        eta = self.eta.eta()
        curr_batch = min(_iter * self._test_batch_size, self._test_set_len)

        print("Testing... [%d/%d (%.2f%%)]" % (
            curr_batch, self._test_set_len, 100 * _iter / self.eta.totiter), "ETA: %.2f min" % eta)

        return outputs

    def validate(self, write=True):
        self.eta.set_epoch(0)
        self.eta.set_totiter(math.ceil(self._val_set_len / self._val_batch_size))

        self.model.eval()

        scores = np.array([.0, .0])
        thresholds = []
        print("Validation starting...")
        for i, data in enumerate(self.val_loader):
            score, threshold = self.__iter_val(i, data)
            scores += score
            thresholds.append(threshold)

        scores_per = 100 * scores / len(self.val_loader)
        self.info.append(['test', [scores[0], scores[1]]])

        average_threshold = np.mean(thresholds)
        if average_threshold > self.model.best_threshold:
            self.model.best_threshold = average_threshold
        if write:
            curr = os.path.join(self.output_path, self.name)
            save = {'type': 'val', 'scores': scores}
            if os.path.isfile(curr + '.npy'):
                prev = np.load(curr + ".npy")
                prev = prev.tolist()
            else:
                prev = []
            prev.append(save)
            np.save(curr, prev)

        print("Percentages of hamming score and f1 score [%.2f %.2f] @ best threshold:%f" %
              (scores_per[0], scores_per[1], self.model.best_threshold))
        print("Validation ended!")

    def __iter_val(self, _iter, data):
        self.eta.update(0, _iter)
        self.eta.start()
        inputs, labels = data

        inputs, targets = self.variable(inputs, labels, requires_grad=False)

        outputs = self.model(inputs)
        if self.cuda:
            outputs = outputs.cpu()

        labels = labels.detach().numpy()
        # best_threshold = find_f2score_threshold(outputs.detach().numpy(), labels, verbose=True)
        best_threshold = 0.5
        predictions = self.predict_with_loss_layer(outputs, best_threshold, True)
        h_score = hamming_score(labels, predictions)
        f1_ = f1_score(labels, predictions, average='samples')

        self.eta.end()
        eta = self.eta.eta()
        curr_batch = min(_iter * self._val_batch_size, self._val_set_len)

        scores = np.array([h_score, f1_])
        score_per = 100 * scores
        print("Validating... [%d/%d (%.2f%%)]" % (curr_batch, self._val_set_len, 100 * _iter / self.eta.totiter),
              "| Hamming score and f1 score = [%.2f %.2f]" % (score_per[0], score_per[1]), "ETA: %.2f min" % eta)

        return scores, best_threshold

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


class Model(object):
    def __init__(self, model, num_labels, parameters, data_set=None):

        self.num_labels = num_labels
        self.model = model
        if data_set is None:
            self.train_data_set = None
            self.val_data_set = None
            self.test_data_set = None

            self.test_data_set = SubRandomDataSetFolder(2)

        self.params = parameters
        self.best_threshold = 0

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def __call__(self, inputs):
        return self.model(inputs)

    def create_datasets(self, **kwargs):
        if 'train' in kwargs:
            self.train_data_set = kwargs['train']
        if 'validation' in kwargs:
            self.val_data_set = kwargs['validation']
        if 'test' in kwargs:
            self.test_data_set = kwargs['test']

    def to_cuda(self):
        self.model = self.model.cuda()

    def to_cpu(self):
        self.model = self.model.cpu()

    def identifier(self):
        bargs = bytes()
        for param in self.params[0:-1]:
            bargs += bytes(str(param), 'utf-8')+b' '
        bargs += bytes(str(self.params[-1]), 'utf-8')

        return bytes(self.__class__.__name__, 'utf-8') + b' ' + bargs

    def save(self, loc):
        torch.save(self.model.state_dict(), loc + ".pt")
        np.save(loc, [self.num_labels, self.best_threshold])

    def adjust_last_layer(self, mode="train", cuda=True):
        raise NotImplementedError("Implement adjust_last_layer method")

    @staticmethod
    def load(args, clazz, loc):
        params = np.load(loc + ".npy")
        num_labels, best_threshold = params[0], params[1]

        model = clazz.models[args[0]](pretrained=False)
        pt = torch.load(loc + ".pt")
        model.load_state_dict(pt)
        model_ = clazz(model, num_labels, args)
        model_.best_threshold = best_threshold
        return model_


class VGGModel(Model):

    vgg_models = {"11": models.vgg11,
                  "11bn": models.vgg11_bn,
                  "13": models.vgg13,
                  "13bn": models.vgg13_bn,
                  "16": models.vgg16,
                  "16bn": models.vgg16_bn,
                  "19": models.vgg19,
                  "19bn": models.vgg19_bn}

    def __init__(self, model, num_labels, parameters, data_set=None):

        super(VGGModel, self).__init__(model, num_labels, parameters, data_set)

    @staticmethod
    def init_11(num_labels=1000, pretrained=True, data_set=None, **kwargs):
        batch_norm = kwargs.get("batch_norm", True)
        if batch_norm:
            model = VGGModel.vgg_models["11bn"](pretrained=pretrained)
            parameters = ["11bn"]
        else:
            model = VGGModel.vgg_models["11"](pretrained=pretrained)
            parameters = ["11"]
        return VGGModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_13(num_labels=1000, pretrained=True, data_set=None, **kwargs):
        batch_norm = kwargs.get("batch_norm", True)
        if batch_norm:
            model = VGGModel.vgg_models["13bn"](pretrained=pretrained)
            parameters = ["13bn"]
        else:
            model = VGGModel.vgg_models["13"](pretrained=pretrained)
            parameters = ["13bn"]
        return VGGModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_16(num_labels=1000, pretrained=True, data_set=None, **kwargs):
        batch_norm = kwargs.get("batch_norm", True)
        if batch_norm:
            model = VGGModel.vgg_models["16bn"](pretrained=pretrained)
            parameters = ["16bn"]
        else:
            model = VGGModel.vgg_models["16"](pretrained=pretrained)
            parameters = ["16"]
        return VGGModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_19(num_labels=1000, pretrained=True, data_set=None, **kwargs):
        batch_norm = kwargs.get("batch_norm", True)
        if batch_norm:
            model = VGGModel.vgg_models["19bn"](pretrained=pretrained)
            parameters = ["19bn"]
        else:
            model = VGGModel.vgg_models["19"](pretrained=pretrained)
            parameters = ["19"]

        return VGGModel(model, num_labels, parameters, data_set)

    def adjust_last_layer(self, mode="train", cuda=True):
        if mode == "train":
            self.num_labels = self.train_data_set.max_class - self.train_data_set.min_class + 1

        old = self.model.classifier
        new = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, self.num_labels)
        )

        if cuda:
            new = new.cuda()

        self.model.classifier = new

        for param1, param2 in zip(self.model.classifier.parameters(), old.parameters()):
            if len(param1) == len(param2):
                param1.data = param2.data


class ResNetModel(Model):

    models = {"18": models.resnet18,
              "34": models.resnet34,
              "50": models.resnet50,
              "101": models.resnet101,
              "152": models.resnet152}

    def __init__(self, model, num_labels, parameters, data_set=None):
        super(ResNetModel, self).__init__(model, num_labels, parameters, data_set)

    @staticmethod
    def init_18(num_labels=1000, pretrained=True, data_set=None):

        model = ResNetModel.models["18"](pretrained=pretrained)
        parameters = ["18"]

        return ResNetModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_34(num_labels=1000, pretrained=True, data_set=None):

        model = ResNetModel.models["34"](pretrained=pretrained)
        parameters = ["34"]

        return ResNetModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_50(num_labels=1000, pretrained=True, data_set=None):

        model = ResNetModel.models["50"](pretrained=pretrained)
        parameters = ["50"]

        return ResNetModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_101(num_labels=1000, pretrained=True, data_set=None):

        model = ResNetModel.models["101"](pretrained=pretrained)
        parameters = ["101"]

        return ResNetModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_152(num_labels=1000, pretrained=True, data_set=None):

        model = ResNetModel.models["152"](pretrained=pretrained)
        parameters = ["152"]

        return ResNetModel(model, num_labels, parameters, data_set)

    def adjust_last_layer(self, mode="train", cuda=True):

        if mode == "train":
            self.num_labels = self.train_data_set.max_class - self.train_data_set.min_class + 1

        if self.num_labels == 1000:
            return
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, self.num_labels)

        if cuda:
            self.model.avgpool = self.model.avgpool.cuda()
            self.model.fc = self.model.fc.cuda()


class DenseNetModel(Model):
    models = {"121": models.densenet121,
              "169": models.densenet161,
              "201": models.densenet169,
              "161": models.densenet201,
              }

    def __init__(self, model, num_labels, parameters, data_set=None):
        super(DenseNetModel, self).__init__(model, num_labels, parameters, data_set)

    @staticmethod
    def init_121(num_labels=1000, pretrained=True, data_set=None):

        model = DenseNetModel.models["121"](pretrained=pretrained)
        parameters = ["121"]

        return DenseNetModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_169(num_labels=1000, pretrained=True, data_set=None):

        model = DenseNetModel.models["169"](pretrained=pretrained)
        parameters = ["169"]

        return DenseNetModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_161(num_labels=1000, pretrained=True, data_set=None):

        model = DenseNetModel.models["161"](pretrained=pretrained)
        parameters = ["161"]

        return DenseNetModel(model, num_labels, parameters, data_set)

    @staticmethod
    def init_201(num_labels=1000, pretrained=True, data_set=None):

        model = DenseNetModel.models["201"](pretrained=pretrained)
        parameters = ["201"]

        return DenseNetModel(model, num_labels, parameters, data_set)

    def adjust_last_layer(self, mode="train", cuda=True):

        if mode == "train":
            self.num_labels = self.train_data_set.max_class - self.train_data_set.min_class + 1

        if self.num_labels == 1000:
            return
        self.model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, self.num_labels)

        if cuda:
            self.model.avgpool = self.model.avgpool.cuda()
            self.model.fc = self.model.fc.cuda()


class GoogLeNetModel(Model):
    models = {"v3": models.inception_v3,
              "": models.inception_v3
              }

    def __init__(self, model, num_labels, parameters, data_set=None):
        super(GoogLeNetModel, self).__init__(model, num_labels, parameters, data_set)

    @staticmethod
    def init_v3(num_labels=1000, pretrained=True, data_set=None):

        model = GoogLeNetModel.models["v3"](pretrained=pretrained)
        parameters = ["18"]

        return ResNetModel(model, num_labels, parameters, data_set)

    def adjust_last_layer(self, mode="train", cuda=True):

        if mode == "train":
            self.num_labels = self.train_data_set.max_class - self.train_data_set.min_class + 1

        if self.num_labels == 1000:
            return

        num_features = self.model.fc.in_features
        self.model.AuxLogits = models.inception.InceptionAux(768, self.num_labels)
        self.model.fc = torch.nn.Linear(num_features, self.num_labels)

        if cuda:
            self.model.AuxLogits = self.model.AuxLogits.cuda()
            self.model.fc = self.model.fc.cuda()


class AlexNetModel(Model):
    models = {"": models.alexnet}

    def __init__(self, model, num_labels, parameters, data_set=None):
        super(AlexNetModel, self).__init__(model, num_labels, parameters, data_set)

    @staticmethod
    def init(num_labels=1000, pretrained=True, data_set=None):

        model = AlexNetModel.models[""](pretrained=pretrained)
        parameters = [""]

        return AlexNetModel(model, num_labels, parameters, data_set)

    def adjust_last_layer(self, mode="train", cuda=True):

        if mode == "train":
            self.num_labels = self.train_data_set.max_class - self.train_data_set.min_class + 1

        old = self.model.classifier
        new = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 6 * 6, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, self.num_labels),
        )

        if cuda:
            new = new.cuda()

        self.model.classifier = new

        for param1, param2 in zip(self.model.classifier.parameters(), old.parameters()):
            if len(param1) == len(param2):
                param1.data = param2.data


class SqueezeNetModel(Model):

    models = {"": models.squeezenet1_1}

    def __init__(self, model, num_labels, parameters, data_set=None):
        super(SqueezeNetModel, self).__init__(model, num_labels, parameters, data_set)

    @staticmethod
    def init(num_labels=1000, pretrained=True, data_set=None):

        model = SqueezeNetModel.models[""](pretrained=pretrained)
        parameters = [""]

        return SqueezeNetModel(model, num_labels, parameters, data_set)

    def adjust_last_layer(self, mode="train", cuda=True):

        if mode == "train":
            self.num_labels = self.train_data_set.max_class - self.train_data_set.min_class + 1

        if self.num_labels == self.model.num_classes:
            return

        num_features = self.model.classifier[1].in_channels

        features = list(self.model.classifier.children())
        features[1] = torch.nn.Conv2d(num_features, self.num_labels, 1)
        # features[3] = torch.nn.AvgPool2d(14, stride=1)

        self.model.num_classes = self.num_labels
        self.model.classifier = torch.nn.Sequential(*features)

        if cuda:
            self.model.classifier = self.model.classifier.cuda()
