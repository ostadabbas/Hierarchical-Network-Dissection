import torch
import torchvision
import os
import random
import csv
import numpy as np
from PIL import Image
import concurrent.futures
import cv2
from facenet_pytorch import MTCNN
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import ImageFile
import math
import settings
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True
from matplotlib import pyplot as plt

class visdict():
    def __init__(self, root, batch_size):

        self.images = []
        self.batch_size = batch_size
        self.mtcnn = MTCNN(image_size=256)

        files = ['Attributes.csv', 'Action_Units.csv', 'Parts_half.csv']

        for file in files:
            with open(root + file, 'r') as f:
                reader = csv.DictReader(f)
                for line in reader:
                    self.images.append(line)

        # print(len(self.images))

        # with open(csv_file, 'r') as file:
        #     reader = csv.DictReader(file)
        #     for line in reader:
        #         self.images.append(line)

        # # self.images = self.images[:320]

        # categories = self.images[0].keys()
        # self.headers = categories
        # self.categories = [x for x in categories if not x == 'image']
        self.num_batches = math.ceil(len(self.images) / self.batch_size)

        self.dense_labels = ['AU_1', 'AU_2', 'AU_4', 'AU_5', 'AU_6', 'AU_9', 'AU_12', 'AU_15', 'AU_17', 'AU_20', 'AU_25', 'AU_26',
                             'nose', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'mouth', 'left_cheek', 'right_cheek',
                             '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bushy_Eyebrows', 'No_Beard', 'Bags_Under_Eyes','Big_Lips',
                             'Big_Nose', 'Double_Chin', 'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Rosy_Cheeks',
                             'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'Pointy_Nose', 'Smiling', 'Wearing_Lipstick']

        self.dense_label_mapping = dict((x, []) for x in self.dense_labels)
        self.ahead = settings.TALLY_AHEAD
        self.queue = []
        self.images_per_label()


    def get_sample(self, idx, crop):

        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_path = self.images[idx]['image']
        img = Image.open('visual_dictionary/' + img_path)
        img = img.resize((settings.INPUT_SIZE, settings.INPUT_SIZE)) #

        if img.mode == 'L':
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg

        sample = {}
        tr_tensor = transforms.ToTensor()
        tr_PIL = transforms.ToPILImage()

        if crop:
            cropped = self.mtcnn(img.copy())
            crop_points = self.mtcnn.detect(img.copy(), landmarks=False)
            cp = crop_points[0]
            NoneType = type(None)
            if isinstance(cp, NoneType):
                cp = np.array([0, 0, settings.INPUT_SIZE, settings.INPUT_SIZE]) #
                img = img.resize((settings.INPUT_SIZE, settings.INPUT_SIZE))
                img = tr_tensor(img)
            else:
                cp = cp[0]
            # sample['crop_points'] = cp
                img = cropped
                img = tr_PIL(img)
                img = img.resize((settings.INPUT_SIZE, settings.INPUT_SIZE))
                img = tr_tensor(img)
        else:
            img = tr_tensor(img)

        # sample = dict((x, []) for x in self.headers)
        sample['image'] = img
        sample['i'] = idx
        sample['labels'] = {}
        sample['fn'] = img_path
        curr_labels = self.images[idx]['labels'].split(';')
        # print(img_path)

        for path in curr_labels:
            label = Image.open('visual_dictionary/' + path).convert("L")
            for dl in self.dense_labels:
                if dl in path:
                    subcat = dl
            label = label.resize((settings.INPUT_SIZE, settings.INPUT_SIZE), Image.NEAREST) #
            # print(cp)
            if crop:
                label = label.crop((cp[0], cp[1], cp[2], cp[3]))
                label = label.resize((settings.INPUT_SIZE, settings.INPUT_SIZE), Image.NEAREST)
            # label.show()
            label = tr_tensor(label)
            sample['labels'][subcat] = label
            # sample['subcats'][cat].append(subcat)
        # print(sample, type(sample))

        return sample

    def get_image(self, idx, crop):

        if torch.is_tensor(idx):
            idx = idx.to_list()

        tr_tensor = transforms.ToTensor()
        tr_PIL = transforms.ToPILImage()

        img_path = self.images[idx]['image']
        # print(img_path)
        img = Image.open('visual_dictionary/' + img_path)
        img = img.resize((settings.INPUT_SIZE, settings.INPUT_SIZE)) #

        if img.mode == 'L':
            rgbimg = Image.new("RGB", img.size)
            rgbimg.paste(img)
            img = rgbimg

        if crop:
            cropped = self.mtcnn(img.copy())
            crop_points = self.mtcnn.detect(img.copy(), landmarks=False)
            cp = crop_points[0]
            NoneType = type(None)
            if isinstance(cp, NoneType):
                img = img.resize((settings.INPUT_SIZE, settings.INPUT_SIZE))
                img = tr_tensor(img)
            else:
                img = cropped
                img = tr_PIL(img)
                img = img.resize((settings.INPUT_SIZE, settings.INPUT_SIZE))
                img = tr_tensor(img)

        else:
            img = tr_tensor(img)

        return img

    # @threadsafe_generator
    def get_batches(self, face_crop=False, only_image=False):

        num_images = len(self.images)
        # print(num_images)

        for idx in range(0, num_images, self.batch_size):
            batch = []
            if face_crop:
                for j in range(idx, min(idx+self.batch_size, num_images)):
                    if only_image:
                        curr_sample = self.get_image(j, face_crop)
                    else:
                        curr_sample = self.get_sample(j, face_crop)
                    batch.append(curr_sample)
            else:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    indexes = range(idx, min(idx + self.batch_size, num_images))
                    crops = [face_crop] * len(indexes)
                    if only_image:
                        results = executor.map(self.get_image, indexes, crops)
                        # curr_sample = self.get_image(j, face_crop)
                        # batch.append(curr_sample)
                    else:
                        results = executor.map(self.get_sample, indexes, crops)
                    for res in results:
                        # print(res)
                        batch.append(res)
            yield batch

    def images_per_label(self):

        print('Mapping labels to images')

        for idx in range(len(self.images)):
            labels = self.images[idx]['labels'].split(';')
            for lab in labels:
                for subcat in self.dense_labels:
                    if subcat in lab:
                        self.dense_label_mapping[subcat].append(idx)


if __name__ == '__main__':

    a = visdict('visual_dictionary/', settings.BATCH_SIZE)
    print(len(a.images))
    dm = a.dense_label_mapping
    counts = dict((x, 0) for x in dm)

    for key in dm:
        counts[key] = len(dm[key])

    # print(counts)
