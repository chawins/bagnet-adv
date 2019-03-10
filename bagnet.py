import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils import model_zoo

from resnet import conv1x1, conv3x3

dir_path = os.path.dirname(os.path.realpath(__file__))

__all__ = ['bagnet9', 'bagnet17', 'bagnet33']

model_urls = {
    'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar',
    'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar',
    'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar',
}


class BottleneckNoBn(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 kernel_size=3):
        super(BottleneckNoBn, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        # self.bn1 = nn.BatchNorm2d(planes)
        if kernel_size == 3:
            self.conv2 = conv3x3(planes, planes, stride)
        else:
            self.conv2 = conv1x1(planes, planes, stride)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        # self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        # out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TinyBagNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 strides=[1, 2, 2, 2],
                 kernel3=[0, 0, 0, 0],
                 num_classes=10,
                 avg_pool=False,
                 patch_size=None,
                 patch_stride=None):

        super(TinyBagNet, self).__init__()
        self.inplanes = 16
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 16, layers[0], stride=strides[0], kernel3=kernel3[0])
        self.layer2 = self._make_layer(
            block, 64, layers[1], stride=strides[1], kernel3=kernel3[1])
        # self.layer3 = self._make_layer(
        #     block, 128, layers[2], stride=strides[2], kernel3=kernel3[2])
        self.fc = nn.Linear(4096, num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride,
                            downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):

        def forward_pass(x):
            x = self.conv1(x)
            # x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            # print(x.size())
            # x = self.layer3(x)
            # x = self.layer4(x)

            if self.avg_pool:
                x = nn.AvgPool2d(x.size(2), stride=1)(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
            else:
                # x = x.permute(0, 2, 3, 1)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
            return x

        if self.patch_size is not None:
            num_patches = x.size(-1) // self.patch_stride + 1
            pad = torch.nn.ZeroPad2d(self.patch_size // 2)
            padded = pad(x)

            outputs_list = []
            for yy in range(num_patches):
                for xx in range(num_patches):
                    y_start = yy * self.patch_stride
                    x_start = xx * self.patch_stride
                    x = padded[:, :, y_start:y_start + self.patch_size,
                               x_start:x_start + self.patch_size]
                    x = forward_pass(x)
                    outputs_list.append(x)
            x = torch.stack(outputs_list, 1)
            x = x.mean(1)
        else:
            x = forward_pass(x)
        return x


class SmallBagNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 strides=[1, 2, 2, 2],
                 kernel3=[0, 0, 0, 0],
                 num_classes=10,
                 avg_pool=True,
                 patch_size=None,
                 patch_stride=None):

        super(SmallBagNet, self).__init__()
        self.inplanes = 64
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        # self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=strides[0], kernel3=kernel3[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=strides[1], kernel3=kernel3[1])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=strides[2], kernel3=kernel3[2])
        # self.layer4 = self._make_layer(
        #     block, 512, layers[3], stride=strides[3], kernel3=kernel3[3])
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride,
                            downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):

        def forward_pass(x):
            x = self.conv1(x)
            x = self.conv2(x)
            # x = self.bn1(x)
            x = self.relu(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            # x = self.layer4(x)

            if self.avg_pool:
                x = nn.AvgPool2d(x.size()[2], stride=1)(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
            else:
                x = x.permute(0, 2, 3, 1)
                x = self.fc(x)
            return x

        if self.patch_size is not None:
            num_patches = x.size(-1) // self.patch_stride + 1
            pad = torch.nn.ZeroPad2d(self.patch_size // 2)
            padded = pad(x)

            outputs_list = []
            # outputs = torch.empty([x.size()[0], num_patches**2, 10],
            #                       dtype=torch.float32, device='cuda')
            for yy in range(num_patches):
                for xx in range(num_patches):
                    y_start = yy * self.patch_stride
                    x_start = xx * self.patch_stride
                    x = padded[:, :, y_start:y_start + self.patch_size,
                               x_start:x_start + self.patch_size]
                    x = forward_pass(x)
                    outputs_list.append(x)
                    # outputs[:, yy * num_patches + xx] = x
            x = torch.stack(outputs_list, 1)
            x = x.mean(1)
        else:
            x = forward_pass(x)
        return x


class BagNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 strides=[1, 2, 2, 2],
                 kernel3=[0, 0, 0, 0],
                 num_classes=1000,
                 avg_pool=True):

        super(BagNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=strides[0], kernel3=kernel3[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=strides[1], kernel3=kernel3[1])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=strides[2], kernel3=kernel3[2])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=strides[3], kernel3=kernel3[3])
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.avg_pool = avg_pool
        self.block = block

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3
        layers.append(block(self.inplanes, planes, stride,
                            downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3
            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.avg_pool:
            x = nn.AvgPool2d(x.size()[2], stride=1)(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.fc(x)

        return x


def bagnet33(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3],
                   strides=strides, kernel3=[1, 1, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet33']))
    return model


def bagnet17(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-17 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3],
                   strides=strides, kernel3=[1, 1, 1, 0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet17']))
    return model


def bagnet9(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3],
                   strides=strides, kernel3=[1, 1, 0, 0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet9']))
    return model
