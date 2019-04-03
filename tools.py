from torch import nn
import torch
from torchvision import utils
import numpy as np
import math
from matplotlib import pyplot as plt

# Configure loss
mse = nn.MSELoss(reduction='sum')
bce = nn.BCELoss(reduction='none')


def G_loss(g1, g2, pixel_label, d, adv_cls_label, trade_off):
    pixel_loss = mse(g1, pixel_label) + mse(g2, pixel_label)
    adv_cls_loss = torch.sum(bce(d, adv_cls_label) * trade_off)
    return pixel_loss + adv_cls_loss

def D_loss(d, adv_cls_label):
    return torch.sum(bce(d, adv_cls_label))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, .0, .1)
        torch.nn.init.constant_(m.bias.data, .0)

def show_grid(x, figsize=(10, 20), nrow=8):
    nums, _, length, width = x.size()
    img_grid = utils.make_grid(x, normalize=True, scale_each=True, nrow=nrow)
    r = math.ceil(nums / nrow)
    plt.figure(figsize=figsize)
    plt.imshow(np.transpose(img_grid.detach().cpu().numpy(), (1, 2, 0)))
    plt.gca().xaxis.set_ticks_position("top")
    plt.xticks(np.arange(0, (nrow+1) * (width+2), width+2), np.arange(nrow))
    plt.yticks(np.arange(0, (r+1) * (length+2), (length+2)), np.arange(r))