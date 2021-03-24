from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from torchvision import transforms

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def loadmodel(fn):

    model = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    # model.conv2d_4b.register_forward_hook(fn)
    # model.repeat_1[4].branch2[2].register_forward_hook(fn)
    # model.repeat_2[9].branch1[2].register_forward_hook(fn)
    model.block8.branch1.register_forward_hook(fn)

    return model.eval()

if __name__ == '__main__':

    mo = loadmodel(hook_feature)
    print(mo)

