import torchvision.models as models
import torch
from torch import nn
import numpy as np
import cv2


features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def loadmodel(fn):

    model = models.resnet50(pretrained=False)

    class Flatten(nn.Module):
        def forward(self, input):
            return input.view(input.size(0), -1)

    model.avgpool = nn.Sequential(Flatten(),
                                  nn.Linear(32768, 2048),
                                  nn.ReLU(),
                                  nn.Linear(2048, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 64),
                                  nn.ReLU())

    model.fc = nn.Linear(64, 1)
    model.load_state_dict(torch.load('model/age_3_val_loss_6.6400.pth'))

    # model.layer1[2].conv1.register_forward_hook(fn)
    # model.layer2[3].conv1.register_forward_hook(fn)
    # model.layer3[2].conv1.register_forward_hook(fn)
    model.layer4[2].conv1.register_forward_hook(fn)
    model.eval()

    return model

if __name__ == '__main__':

    mo = loadmodel(hook_feature)
    print(mo)

