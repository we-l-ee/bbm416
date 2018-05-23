import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import random
import torch
import numpy as np
from utils import label_info
# Some images has alpha channel to. All converted to RGB.


def image_loader(image_path, image_size=224):
    image = Image.open(image_path).convert("RGB")
    # image = np.array(image); image = image/ np.linalg.norm(image); image = Image.fromarray(Image.)
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([#transforms.Resize(imsize),
                                transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalizer])

    return trans(image)


def image_loader_cuda(image_path):
    image = Image.open(image_path)
    image = image.convert("RGB")

    imsize = (224, 224)
    loader = transforms.Compose([transforms.Resize(imsize), transforms.ToTensor()])
    image = loader(image)

    return image.cuda()


class DatasetText(Dataset):
    def __init__(self, path, label_encoder, loader=image_loader):
        _in = open(path, 'r')
        self.data = []
        self.label = []
        self.label_encoder = label_encoder
        self.loader = loader
        self.trained = []

    def load(self, path, root='Images'):
        with open(path,'r') as _in:

            for line in _in.readlines():
                self.data.append(
                    self.loader(os.path.join(root, line.strip()))
                )
                label = os.path.split(os.path.split(line)[0])[1]
                self.label_encoder.encode_label(label)
                self.label.append(
                    self.label_encoder.decodeLabel(label)
                )
            self.len = len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class DataSetFolder(Dataset):
    def __init__(self,  loader=image_loader):
        self.data = []
        self.label = []
        self.loader = loader
        self.trained = []

    def load(self, root, mode="val"):

        for file_name in os.listdir(root):
            self.data.append(self.loader(os.path.join(root,file_name)))
            if mode == "val":
                l = file_name.split("_")[3].split('.')[0]
                labels = [int(i)-1 for i in l[1:-1].split(',')]
            elif mode == "test":
                labels.append(int(file_name))
            self.label.append(labels)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)


class SubRandomDataSetFolder(Dataset):
    def __init__(self, n=1000, loader=image_loader):
        self.data = []
        self.num = n
        self.label = []
        self.loader = loader
        self.trained = []
        self._name = None
        self.num_classes, self.min_class, self.max_class = None, None, None

    def load(self, root):
        self._name = root
        info_label = label_info(self._name + ".json")
        self.num_classes, self.min_class, self.max_class = len(info_label[0]), info_label[1], info_label[2]

        dir_list = os.listdir(root)
        for i in range(self.num):
            fname = random.choice(dir_list)
            self.data.append(self.loader(os.path.join(root, fname)))
            l = fname.split("_")[3].split('.')[0]
            labels = [int(i.strip()) for i in l[1:-1].split(',')]

            temp_labels = np.zeros(self.max_class - self.min_class + 1)
            for label in labels:
                temp_labels[label-self.min_class] = 1.0
            self.label.append(torch.from_numpy(temp_labels))

    def get_root(self):
        return self._name

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)



