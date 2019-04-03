import argparse
from tensorboardX import SummaryWriter

import os
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from models import Generator, Discriminator
from tools import G_loss, D_loss, weights_init_normal
from dataset import arrange_data, WIDER

writer = SummaryWriter()
os.makedirs('saved_models', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default="./WIDER/WIDER_train/images/", help='path of the training dataset')
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=6, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=3, help='epoch from which to start lr decay')
parser.add_argument('--alpha', type=float, default=0.0001, help='weight trade-off parameter')
parser.add_argument('--beta', type=float, default=0.001, help='weight trade-off parameter')
parser.add_argument('--hr_height', type=int, default=128, help='size of high res. image height')
parser.add_argument('--hr_width', type=int, default=128, help='size of high res. image width')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling of images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
opt = parser.parse_args()
print(opt)


# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

# Load pretrained generator
weights_ = torch.load("./models/generator_40.pth")
weights = OrderedDict()
for k, v in weights_.items():
    weights[k.split('module.')[-1]] = v
generator.load_state_dict(weights)

# Initialize fc layer of discriminator
discriminator.apply(weights_init_normal)

# find gpu devices
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_nums = torch.cuda.device_count()

# Use all GPUs by default
if device_nums > 1:
    generator = torch.nn.DataParallel(generator, device_ids=range(device_nums))
    discriminator = torch.nn.DataParallel(discriminator, device_ids=range(device_nums))

# Set models to gpu
generator = generator.to(device)
discriminator = discriminator.to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# learning rate schedule
scheduler_G = StepLR(optimizer_G, step_size=3, gamma=0.1)
scheduler_D = StepLR(optimizer_D, step_size=3, gamma=0.1)

# prepare trade off parameters
trade_off = torch.FloatTensor([opt.alpha, opt.beta])

# use generated face to fool discriminator and make generated face easy to be recognized
fake_gen_face = torch.ones((opt.batch_size, 2))

# prepare valid labels to train discriminator
valid_gen_face = torch.ones_like(fake_gen_face) * torch.FloatTensor([0, 1]) # 0, 1
valid_gen_background = torch.zeros_like(fake_gen_face) # 0, 0
valid_ground_face = torch.ones_like(fake_gen_face) # 1, 1
valid_ground_background = torch.ones_like(fake_gen_face) * torch.FloatTensor([1, 0])# 1, 0

# set them to gpus
trade_off = trade_off.to(device)
fake_gen_face = fake_gen_face.to(device)
valid_gen_face = valid_gen_face.to(device)
valid_gen_background = valid_gen_background.to(device)
valid_ground_face = valid_ground_face.to(device)
valid_ground_background = valid_ground_background.to(device)

# prepare data
anno_path = "./WIDER/wider_face_split/wider_face_train_bbx_gt.txt"
path, bbxs = arrange_data(anno_path)
wider = WIDER(opt.train_path, path, bbxs, high_resolution=(opt.hr_height, opt.hr_width))
dataloader = DataLoader(wider, batch_size=opt.batch_size, shuffle=True, num_workers=8)


# ----------
#  Training
# ----------

for epoch in range(opt.epoch, opt.n_epochs):
    scheduler_G.step()
    scheduler_D.step()
    loss_D_ = 0
    loss_G_ = 0
    for i, imgs in enumerate(dataloader):

        # Configure model input
        lr_face = imgs["lr_face"].to(device)
        hr_face = imgs["hr_face"].to(device)
        lr_background = imgs["lr_background"].to(device)
        hr_background = imgs["hr_background"].to(device)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        g1, gen_face = generator(lr_face)
        loss_g = G_loss(g1, gen_face, hr_face, discriminator(gen_face), fake_gen_face, trade_off) / opt.batch_size

        loss_g.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of gan and classification
        loss_d = D_loss(discriminator(gen_face.detach()), valid_gen_face) + \
                 D_loss(discriminator(generator(lr_background)[1].detach()), valid_gen_background) + \
                 D_loss(discriminator(hr_face), valid_ground_face) +\
                 D_loss(discriminator(hr_background), valid_ground_background)
        loss_d /= (4 * opt.batch_size)

        loss_d.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                                                            (epoch, opt.n_epochs, i, len(dataloader),
                                                            loss_d.item(), loss_g.item()))

        writer.add_scalars('data/loss_iter', {"loss_D": loss_d.item(), "loss_G": loss_g.item()}, i)

        loss_D_ += loss_d.item()
        loss_G_ += loss_g.item()
        batches_done = epoch * len(dataloader) + i

    writer.add_scalars('data/loss_epoch', {"loss_D": loss_D_, "loss_G": loss_G_}, epoch)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), 'saved_models/generator_%d.pth' % epoch)
        torch.save(discriminator.state_dict(), 'saved_models/discriminator_%d.pth' % epoch)

writer.export_scalars_to_json("./all_scalars.json")
writer.close()