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
                              # nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Linear(2048, 512),
                              nn.ReLU(),
                              # nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                              nn.Linear(512, 64),
                              nn.ReLU())
                              # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.fc = nn.Sequential(nn.Linear(64, 2))
    model.load_state_dict(torch.load('model/gender_1_val_acc_0.9082.pth'))

    # model.layer1[2].conv1.register_forward_hook(fn)
    # model.layer2[2].conv1.register_forward_hook(fn)
    # model.layer3[2].conv1.register_forward_hook(fn)
    model.layer4[2].conv1.register_forward_hook(fn)

    return model.eval()

if __name__ == '__main__':

    mo = loadmodel(hook_feature)
    print(mo)

