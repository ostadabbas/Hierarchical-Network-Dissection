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
    model.fc = nn.Sequential(nn.Linear(2048, 2))
    model.cuda()
    model.load_state_dict(torch.load('model/smile_nonsmile_0.912.pth'))

    # model.layer1[2].conv1.register_forward_hook(fn)
    # model.layer2[2].conv1.register_forward_hook(fn)
    # model.layer3[2].conv1.register_forward_hook(fn)
    model.layer4[2].conv1.register_forward_hook(fn)
    model.eval()

    return model


if __name__ == '__main__':

    mo = loadmodel(hook_feature)
    print(mo)
