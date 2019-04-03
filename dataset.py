import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from os.path import join
import numpy as np


# Transforms for low resolution images and high resolution images
def transform_hl_pair(hr_height, hr_width):

    lr_transforms = [transforms.Resize((hr_height // 4, hr_width // 4), Image.BICUBIC),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    hr_transforms = [transforms.Resize((hr_height, hr_width), Image.BICUBIC),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(lr_transforms), transforms.Compose(hr_transforms)

def arrange_data(path):

    with open(path, 'r') as f:
        data = f.readlines()

    data = [x.strip() for x in data]
    flags = []
    for (i, x) in enumerate(data):
        if (x.endswith('.jpg')):
            flags.append(i)
        else:
            data[i] = [int(loc) for loc in x.split(' ')[:4]]

    path = np.array(data)[flags].tolist()
    bbxs = [x[2:] for x in np.split(data, flags[1:])]
    return path, bbxs


def iou(a, b):
    sizea = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    sizeb = (b[2] - b[0]) * (b[3] - b[1])
    tl = np.maximum(a[:, :2], b[:2])
    br = np.minimum(a[:, 2:], b[2:])
    wh = np.maximum(br - tl, 0)
    size = wh[:, 0] * wh[:, 1]
    return size / (sizea + sizeb - size)


class WIDER(Dataset):

    def __init__(self, base, path, bbxs, high_resolution=(128, 128)):
        self.base = base
        self.path = path
        self.bbxs = bbxs
        self.lr, self.hr = transform_hl_pair(*high_resolution)

    def __len__(self):
        return len(self.path)

    def __getitem__(self, idx):
        img = Image.open(join(self.base, self.path[idx]))
        bbxs = np.vstack(self.bbxs[idx])
        # random select one face
        idx = np.random.randint(0, len(bbxs), 1)
        bbxs[:, 2:] += bbxs[:, 0:2]

        bbx = bbxs[idx, :].squeeze()
        true = img.crop(bbx)
        # random crop a fix-sized background patch
        x, y = np.random.randint(0, min(img.size) - 128, 2)
        bg = [x, y, x + 128, y + 128]
        if np.all(iou(bbxs, bg) < 0.5):
            false = img.crop(bg)
        else:
            false = Image.fromarray(np.random.randint(0, 256, size=(128, 128, 3)).astype('uint8'))
            print("use random noise.")
        return {"lr_face": self.lr(true), "lr_background": self.lr(false),
                "hr_face": self.hr(true), "hr_background": self.hr(false)}


if __name__ == '__main__':
    train_path = "./WIDER/WIDER_train/images/"
    path, bbxs = arrange_data()
    wider = WIDER(train_path, path, bbxs)
    result = wider[22]