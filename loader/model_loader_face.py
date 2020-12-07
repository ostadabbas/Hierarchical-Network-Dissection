from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
from torchvision import transforms

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


def loadmodel(fn):

    model = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    model.block8.branch1.register_forward_hook(fn)
    # model.conv2d_4b.register_forward_hook(fn)
    # model.repeat_1[4].branch2[2].register_forward_hook(fn)
    # model.repeat_2[9].branch1[2].register_forward_hook(fn)

    return model.eval()

if __name__ == '__main__':

    # img = cv2.imread('../visual_dictionary/AU_12/SN001_3/original.png')[:,:,::-1]
    # img = cv2.resize(img, (224, 224))
    # img2 = cv2.imread('../face_data/malekumar_env03/headrende0008.png')[:,:,::-1]
    # print(img.shape)
    # print(img.min(), img.max())
    # mtcnn = MTCNN(image_size=224)
    # cropped = mtcnn(img.copy())
    # print(cropped.shape)
    # print(cropped.min(), cropped.max(), cropped.shape, type(cropped))
    # tr = transforms.ToTensor()
    # tr2 = transforms.ToPILImage()
    # cropped1 = tr2(cropped)
    # cropped1 = cropped1.resize((224, 224))
    # cropped1 = tr(cropped1)
    # cropped2 = mtcnn(img2.copy())
    mo = loadmodel(hook_feature)
    print(mo)
    # embedding1 = mo(cropped.unsqueeze(0).cuda())
    # # embedding2 = mo(cropped2.unsqueeze(0).cuda())
#
    # print(features_blobs[0].shape)
