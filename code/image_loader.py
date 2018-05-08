import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os


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
    def __init__(self, path, label_encoder, root='Images', loader=image_loader):
        _in = open(path, 'r')
        self.data = []
        self.label = []

        self.trained = []

        for line in _in.readlines():
            self.data.append(
                loader(os.path.join(root, line.strip()))
            )
            label = os.path.split(os.path.split(line)[0])[1]
            label_encoder.encode_label(label)
            self.label.append(
                label_encoder.decodeLabel(label)
            )
        self.len = len(self.data)

    def __getitem__(self, item):
        return self.data[item], self.label[item]

    def __len__(self):
        return self.len