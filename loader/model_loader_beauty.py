import torchvision.models as models
import torch
import numpy as np
import cv2
import torch.nn as nn
import math
from collections import OrderedDict

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def loadmodel(fn):

    def load_model(pretrained_dict, new):
        model_dict = new.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        new.load_state_dict(model_dict)

    def conv3x3(in_planes, out_planes, stride=1):
        # 3x3 convolution with padding
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            m = OrderedDict()
            m['conv1'] = conv3x3(inplanes, planes, stride)
            m['bn1'] = nn.BatchNorm2d(planes)
            m['relu1'] = nn.ReLU(inplace=True)
            m['conv2'] = conv3x3(planes, planes)
            m['bn2'] = nn.BatchNorm2d(planes)
            self.group1 = nn.Sequential(m)
            self.relu = nn.Sequential(nn.ReLU(inplace=True))
            self.downsample = downsample

        def forward(self, x):
            if self.downsample is not None:
                residual = self.downsample(x)
            else:
                residual = x
            out = self.group1(x) + residual
            out = self.relu(out)

            return out

    class ResNet(nn.Module):
        def __init__(self, block, layers, num_classes=1000):
            self.inplanes = 64
            super(ResNet, self).__init__()

            m = OrderedDict()
            m['conv1'] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            m['bn1'] = nn.BatchNorm2d(64)
            m['relu1'] = nn.ReLU(inplace=True)
            m['maxpool'] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.group1= nn.Sequential(m)

            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

            self.avgpool = nn.Sequential(nn.AvgPool2d(7))
            self.group2 = nn.Sequential(
                OrderedDict([
                    ('fullyconnected', nn.Linear(512 * block.expansion, num_classes))
                ])
            )

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight.data)
                    torch.nn.init.constant_(m.bias.data, 0)


        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.group1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.group2(x)

            return x

    net = ResNet(block = BasicBlock, layers = [2, 2, 2, 2], num_classes = 1).cuda()
    load_model(torch.load('model/resnet18.pth', encoding='latin1'), net)
    # net.layer1[1].group1.conv2.register_forward_hook(fn)
    # net.layer2[1].group1.conv2.register_forward_hook(fn)
    # net.layer3[1].group1.conv2.register_forward_hook(fn)
    net.layer4[1].group1.conv2.register_forward_hook(fn)
    return net.eval()

if __name__ == '__main__':

    mo = loadmodel(hook_feature)
    print(mo)
