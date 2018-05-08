import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import random

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
        return self.len

class DatasetFolderSub(Dataset):
    def __init__(self, path, n = 1000, loader=image_loader):
        _in = open(path, 'r')
        self.data = []
        self.num = n
        self.label = []
        self.loader = image_loader
        self.trained = []


    def load(self, root):
        for i in range(self.num):
            dir = random.choice(os.listdir(root))
            self.data.append(self.loader(dir))
            _,fname = os.path.split(dir)
            l = fname.split("_")[3]
            labels = [int(i) for i in l[1:-1].split(',')]
            self.label.append(labels)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.len

