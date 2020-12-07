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

    '''
    model.avgpool = nn.Sequential(Flatten(),
                                  nn.Linear(32768, 2048),
                                  nn.ReLU(),
                                  nn.Linear(2048, 512),
                                  nn.ReLU(),
                                  nn.Linear(512, 64),
                                  nn.ReLU())

    model.fc = nn.Sequential(nn.Linear(64, 2))
    '''
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
    model.layer4[2].conv1.register_forward_hook(fn)
    # model.layer4[2].conv1.register_forward_hook(fn)

    return model.eval()

if __name__ == '__main__':

    # img = cv2.imread('../visual_dictionary/AU/SN001_3/original.png')
    # img_copy = img.copy()
    # img = cv2.resize(img, (128, 128))
    # img = torch.FloatTensor(np.expand_dims(img.transpose(2, 0, 1), axis=0))
    # print(img.shape)
    # img2 = cv2.imread('../face_data/malekumar_env03/headrende0008.png')[:,:,::-1]
    # print(img.shape)
    # mtcnn = MTCNN(image_size=256)
    # cropped1 = mtcnn(img.copy())
    # cropped2 = mtcnn(img2.copy())
    mo = loadmodel(hook_feature)
    print(mo)
    # embedding = mo(img.cuda())
    # maps = features_blobs[0]
    # print(maps.shape)
    # for i in range(256):
    #     cv2.imshow('i', m)
    #     cv2.waitKey(0)
    # print(embedding)
    # cv2.imshow('i', img_copy)
    # cv2.waitKey(0)
    # embedding2 = mo(cropped2.unsqueeze(0).cuda())
    # print(mo.layer3[1].conv2)
