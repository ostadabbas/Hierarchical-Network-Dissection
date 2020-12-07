import torchvision.models as models
import torch
import numpy as np
import cv2
import torch.nn as nn
import math
from collections import OrderedDict
from torchvision.models import resnet50

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def loadmodel(fn):

    class Flatten(nn.Module):

        def forward(self, x):
            return x.view(x.size(0), -1)


    class FaceNetModel(nn.Module):
        def __init__(self, pretrained=False):
            super(FaceNetModel, self).__init__()

            self.model = resnet50(pretrained)
            embedding_size = 128
            num_classes = 4294
            self.cnn = nn.Sequential(
                self.model.conv1,
                self.model.bn1,
                self.model.relu,
                self.model.maxpool,
                self.model.layer1,
                self.model.layer2,
                self.model.layer3,
                self.model.layer4)

            # modify fc layer based on https://arxiv.org/abs/1703.07737
            self.model.fc = nn.Sequential(
                Flatten(),
                # nn.Linear(100352, 1024),
                # nn.BatchNorm1d(1024),
                # nn.ReLU(),
                nn.Linear(32768, embedding_size))# 100352

            self.model.classifier = nn.Linear(embedding_size, num_classes)

        def l2_norm(self, input):
            input_size = input.size()
            buffer = torch.pow(input, 2)
            normp = torch.sum(buffer, 1).add_(1e-10)
            norm = torch.sqrt(normp)
            _output = torch.div(input, norm.view(-1, 1).expand_as(input))
            output = _output.view(input_size)
            return output

        def freeze_all(self):
            for param in self.model.parameters():
                param.requires_grad = False

        def unfreeze_all(self):
            for param in self.model.parameters():
                param.requires_grad = True

        def freeze_fc(self):
            for param in self.model.fc.parameters():
                param.requires_grad = False

        def unfreeze_fc(self):
            for param in self.model.fc.parameters():
                param.requires_grad = True

        def freeze_only(self, freeze):
            for name, child in self.model.named_children():
                if name in freeze:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True

        def unfreeze_only(self, unfreeze):
            for name, child in self.model.named_children():
                if name in unfreeze:
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False

        # returns face embedding(embedding_size)
        def forward(self, x):
            x = self.cnn(x)
            # print(x.shape)
            x = self.model.fc(x)

            features = self.l2_norm(x)
            # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
            alpha = 10
            features = features * alpha
            return features

        def forward_classifier(self, x):
            features = self.forward(x)
            res = self.model.classifier(features)
            return res

    network = FaceNetModel()
    checkpoint = torch.load('model/best_state_43.pth')
    network.load_state_dict(checkpoint['state_dict'])
    network.model.layer4[1].conv1.register_forward_hook(fn)
    # network.model.layer2[2].conv1.register_forward_hook(fn)
    return network.eval()

if __name__ == '__main__':

    # img = cv2.imread('../visual_dictionary/AU_12/SN001_3/original.png')
    # img_copy = img.copy()
    # img = cv2.resize(img, (112, 112))
    # img = torch.FloatTensor(np.expand_dims(img.transpose(2, 0, 1), axis=0)).cuda()
    # print(img.shape)
    # img2 = cv2.imread('../face_data/malekumar_env03/headrende0008.png')[:,:,::-1]
    # print(img.shape)
    # mtcnn = MTCNN(image_size=256)
    # cropped1 = mtcnn(img.copy())
    # cropped2 = mtcnn(img2.copy())
    mo = loadmodel(hook_feature)
    # embedding = mo(img)
    # maps = features_blobs[0]
    # print(maps.shape)
    print(mo)
    # print(mo)
    # for i in range(256):
    #     cv2.imshow('i', m)
    #     cv2.waitKey(0)
    # print(features_blobs[0].shape)
    # cv2.imshow('i', img_copy)
    # cv2.waitKey(0)
    # embedding2 = mo(cropped2.unsqueeze(0).cuda())
    # print(mo.layer3[1].conv2)
