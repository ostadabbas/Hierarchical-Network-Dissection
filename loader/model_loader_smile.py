import numpy as np
import random
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import models, transforms
import os
from tqdm import tqdm

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
# print(model)
def loadmodel(fn):

    model = models.resnet50(pretrained=True)

    # model.avgpool = nn.Sequential(Flatten(),
    #                               nn.Linear(51200, 2048), # 100352
    #                               nn.ReLU(),
    #                               nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #                               nn.Linear(2048, 512),
    #                               nn.ReLU(),
    #                               nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    #                               nn.Linear(512, 64),
    #                               nn.ReLU(),
    #                               nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.fc = nn.Sequential(nn.Linear(2048, 2))
                             # nn.LogSoftmax(dim=1))
    model.cuda()
    model.load_state_dict(torch.load('model/smile_nonsmile_0.912.pth'))

    model.layer4[2].conv1.register_forward_hook(fn)
    # model.layer2[2].conv1.register_forward_hook(fn)
    model.eval()

    return model

