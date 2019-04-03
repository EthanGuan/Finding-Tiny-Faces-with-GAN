from torch import nn
from torchvision.models import vgg19


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def devonc3x3(in_planes, out_planes, stride):
    return nn.ConvTranspose2d(in_planes, out_planes, 3, stride=stride, padding=1)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class UpSample(nn.Module):
    def __init__(self, in_channels=3, n_res_block=8):
        super(UpSample, self).__init__()
        res_blocks = [BasicBlock(64, 64) for _ in range(n_res_block)]
        upsample4x = [
            devonc3x3(64, 256, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            devonc3x3(256, 256, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.conv1 = nn.Sequential(conv3x3(in_channels, 64),  nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.upsample4x = nn.Sequential(*upsample4x)
        self.conv2 = nn.Sequential(conv1x1(256, 3),  nn.BatchNorm2d(3), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        for layer in self.upsample4x:
            if isinstance(layer, nn.ConvTranspose2d):
                x = layer(x, output_size=[x.size(2)*2, x.size(3)*2])
            else:
                x = layer(x)
        x = self.conv2(x)
        return x


class Refinement(nn.Module):
    def __init__(self, in_channels=3, n_res_block=8):
        super(Refinement, self).__init__()
        res_blocks = [BasicBlock(64, 64) for _ in range(n_res_block)]
        self.conv1 = nn.Sequential(conv3x3(in_channels, 64), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
        self.res_blocks = nn.Sequential(*res_blocks)
        self.conv2 = nn.Sequential(conv3x3(64, 64),  nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
        self.conv3 = nn.Sequential(conv3x3(64, 256), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.conv4 = nn.Sequential(conv3x3(256, 256), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.conv5 = nn.Sequential(conv3x3(256, 3), nn.Tanh())

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3):
        super(Generator, self).__init__()
        self.upsample = UpSample(in_channels, 8)
        self.refinement = Refinement(in_channels, 8)

    def forward(self, x):
        x1 = self.upsample(x)
        x2 = self.refinement(x1)
        return x1, x2


class Discriminator(nn.Module):
        
    def __init__(self):
        super(Discriminator, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = vgg19_model.features[:-1]
        #self.avgpool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        self.fc_gan_cls = nn.Linear(8*8*512, 2)
        self.sigmoid = nn.Sigmoid()




    def forward(self, x):
        x = self.feature_extractor(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_gan_cls(x)
        x = self.sigmoid(x)
        return x
