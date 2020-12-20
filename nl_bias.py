import torch
from torch import nn
import numpy as np
import settings
# from matplotlib import pyplot as plt
import pickle
import os
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


class non_localizable_dict():

    def __init__(self, root_folder):

        self.root = root_folder
        self.age = {}
        self.gender = {}
        self.ethnicity = {}
        self.skin = {}

    def read_age(self):

        groups = ['zero_to_twenty', 'twenty_to_forty', 'forty_to_sixty', 'sixty_plus']

        for age_grp in groups:
            self.age[age_grp] = []
            for file in os.listdir(self.root + '/Age/{}'.format(age_grp)):
                self.age[age_grp].append(self.root + '/Age/{}/{}'.format(age_grp, file))

    def read_gender(self):

        genders = ['Male', 'Female']

        for gen in genders:
            self.gender[gen] = []
            for file in os.listdir(self.root + '/Gender/{}'.format(gen)):
                self.gender[gen].append(self.root + '/Gender/{}/{}'.format(gen, file))

    def read_ethnic(self):

        races = ['white', 'black', 'asian', 'indian']

        for race in races:
            self.ethnicity[race] = []
            for file in os.listdir(self.root + '/Ethnicity/{}'.format(race)):
                self.ethnicity[race].append(self.root + '/Ethnicity/{}/{}'.format(race, file))

    def read_skin(self):

        tones = ['bright', 'dark']

        for tone in tones:
            self.skin[tone] = []
            for file in os.listdir(self.root + '/skin_tone/{}'.format(tone)):
                self.skin[tone].append(self.root + '/skin_tone/{}/{}'.format(tone, file))

features_blobs = []
trf = transforms.ToTensor()

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

data = non_localizable_dict('visual_dictionary/non_localizable')

def age_features(size, curr_task, curr_model):

    data.read_age()
    images = data.age
    # groups = list(images.keys())
    feature_dict = {}

    for grp in images:
        print(grp)
        image_list = images[grp]
        first = True
        for idx, imgpath in tqdm(enumerate(image_list)):

            del features_blobs[:]
            img = Image.open(imgpath)
            img = img.resize((size, size))
            img = trf(img).cuda().unsqueeze(0)
            out = curr_model(img)

            curr_activation = features_blobs[0]

            if first:
                num_units = curr_activation.shape[1]
                h, w = curr_activation.shape[2:]
                features = np.zeros((1000, num_units, h, w))
                first = False

            features[idx:idx+1] = curr_activation

        feature_dict[grp] = features

    np.savez('{}/{}_model_bias(age).npz'.format(settings.NL_FOLDER, curr_task), zero=feature_dict['zero_to_twenty'],
                                                                                twenty=feature_dict['twenty_to_forty'],
                                                                                forty=feature_dict['forty_to_sixty'],
                                                                                sixty=feature_dict['sixty_plus'])


def gender_features(size, curr_task, curr_model):

    data.read_gender()
    images = data.gender
    # groups = list(images.keys())
    feature_dict = {}

    for grp in images:
        print(grp)
        image_list = images[grp]
        first = True
        for idx, imgpath in tqdm(enumerate(image_list)):

            del features_blobs[:]
            img = Image.open(imgpath)
            img = img.resize((size, size))
            img = trf(img).cuda().unsqueeze(0)
            out = curr_model(img)

            curr_activation = features_blobs[0]

            if first:
                num_units = curr_activation.shape[1]
                h, w = curr_activation.shape[2:]
                features = np.zeros((2000, num_units, h, w))
                first = False

            features[idx:idx+1] = curr_activation

        feature_dict[grp] = features

    np.savez('{}/{}_model_bias(gender).npz'.format(settings.NL_FOLDER, curr_task),
                                                   male=feature_dict['Male'],
                                                   female=feature_dict['Female'])


def ethnic_features(size, curr_task, curr_model):

    data.read_ethnic()
    images = data.ethnicity
    # groups = list(images.keys())
    feature_dict = {}

    for grp in images:
        print(grp)
        image_list = images[grp]
        first = True
        for idx, imgpath in tqdm(enumerate(image_list)):

            del features_blobs[:]
            img = Image.open(imgpath)
            img = img.resize((size, size))
            img = trf(img).cuda().unsqueeze(0)
            out = curr_model(img)

            curr_activation = features_blobs[0]

            if first:
                num_units = curr_activation.shape[1]
                h, w = curr_activation.shape[2:]
                features = np.zeros((1000, num_units, h, w))

            features[idx:idx+1] = curr_activation

        feature_dict[grp] = features

    np.savez('{}/{}_model_bias(ethnic).npz'.format(settings.NL_FOLDER, curr_task), white=feature_dict['white'],
                                                                                   black=feature_dict['black'],
                                                                                   asian=feature_dict['asian'],
                                                                                   indian=feature_dict['indian'])

def skin_features(size, curr_task, curr_model):

    data.read_skin()
    images = data.skin
    # groups = list(images.keys())
    feature_dict = {}

    for grp in images:
        print(grp)
        image_list = images[grp]
        first = True
        for idx, imgpath in tqdm(enumerate(image_list)):

            del features_blobs[:]
            img = Image.open(imgpath)
            img = img.resize((size, size))
            img = trf(img).cuda().unsqueeze(0)
            out = curr_model(img)

            curr_activation = features_blobs[0]

            if first:
                num_units = curr_activation.shape[1]
                h, w = curr_activation.shape[2:]
                features = np.zeros((1500, num_units, h, w))
                first = False

            features[idx:idx+1] = curr_activation

        feature_dict[grp] = features

    np.savez('{}/{}_model_bias(skin_tone).npz'.format(settings.NL_FOLDER, curr_task), bright=feature_dict['bright'], dark=feature_dict['dark'])


# age_features()
# gender_features()
# ethnic_features()

def bias_analysis():

    for task in tqdm(tasks):
        print(task)

        # og_tally = np.load('{}/{}_layer4_tally20.npz'.format(settings.OUTPUT_FOLDER, task))['tally']
        # num_units = og_tally.shape[0]
        # unit_info = {}
        # unit_prob_tally = {}

        # for unit in range(num_units):
        #     unit_scores = og_tally[unit]
        #     max_iou = np.max(unit_scores)
        #     if max_iou > 0.04:
        #         unit_info[unit] = 'localizable'
        #     else:
        #         unit_info[unit] = 'unlocalizable'

        age_features = np.load('{}/{}_model_bias(age)_layer2.npz'.format(settings.NL_FOLDER, task))
        gender_features = np.load('{}/{}_model_bias(gender)_layer2.npz'.format(settings.NL_FOLDER, task))
        ethnic_features = np.load('{}/{}_model_bias(ethnic)_layer2.npz'.format(settings.NL_FOLDER, task))
        skin_features = np.load('{}/{}_model_bias(skin_tone)_layer2.npz'.format(settings.NL_FOLDER, task))
        # print(list(skin_features.keys()))

        zero = age_features['zero']
        twenty = age_features['twenty']
        forty = age_features['forty']
        sixty = age_features['sixty']

        male = gender_features['male']
        female = gender_features['female']

        white = ethnic_features['white']
        black = ethnic_features['black']
        asian = ethnic_features['asian']
        indian = ethnic_features['indian']

        bright = skin_features['male']
        dark = skin_features['female']

        all_age_maps = np.concatenate([zero, twenty, forty, sixty], axis=0)
        all_gender_maps = np.concatenate([male, female], axis=0)
        all_ethnic_maps = np.concatenate([white, black , asian, indian], axis=0)
        all_skin_maps = np.concatenate([bright, dark], axis=0)

        print(all_age_maps.shape, all_gender_maps.shape, all_ethnic_maps.shape, all_skin_maps.shape)

        age_indexes = {'zero_to_twenty':[0, 1000], 'twenty_to_forty':[1000, 2000], 'forty_to_sixty':[2000, 3000], 'sixty_plus':[3000, 4000]}
        gender_indexes = {'male':[0, 2000], 'female':[2000, 4000]}
        ethnic_indexes = {'white':[0, 1000], 'black':[1000, 2000], 'asian':[2000, 3000], 'indian':[3000, 4000]}
        skin_indexes = {'bright':[0, 1500], 'dark':[1500, 3000]}

        count = 0

        # for unit_idx in range(num_units):

            # if count == 3:
            #     break

            # count += 1

        # unit_prob_tally[5] = {'age':{}, 'gender':{}, 'ethnic':{}} #
        unit_prob_tally = {}

        for ux in range(num_units):

            unit_prob_tally[ux+1] = {'age':{}, 'gender':{}, 'ethnic':{}, 'skin':{}}
            # print(ux)
            age_scores = dict((x, 0) for x in age_indexes)
            gender_scores = dict((x, 0) for x in gender_indexes)
            ethnic_scores = dict((x, 0) for x in ethnic_indexes)
            skin_scores = dict((x, 0) for x in skin_indexes)

            unit_age_maps = all_age_maps[:, ux, :, :] #
            age_min, age_max = unit_age_maps.min(), unit_age_maps.max()
            unit_gender_maps = all_gender_maps[:, ux, :, :] #
            gender_min, gender_max = unit_gender_maps.min(), unit_gender_maps.max()
            unit_ethnic_maps = all_ethnic_maps[:, ux, :, :] #
            ethnic_min, ethnic_max = unit_ethnic_maps.min(), unit_ethnic_maps.max()
            unit_skin_maps = all_skin_maps[:, ux, :, :] #
            skin_min, skin_max = unit_skin_maps.min(), unit_skin_maps.max()

            sorted_age_indexes = np.argsort(np.max(all_age_maps.reshape(4000, -1), axis=-1))
            sorted_gender_indexes = np.argsort(np.max(all_gender_maps.reshape(4000, -1), axis=-1))
            sorted_ethnic_indexes = np.argsort(np.max(all_ethnic_maps.reshape(4000, -1), axis=-1))
            sorted_skin_indexes = np.argsort(np.max(all_skin_maps.reshape(3000, -1), axis=-1))

            for i, idx in enumerate(sorted_age_indexes):
                rank = (i+1) / 4000
                fmap_score = (unit_age_maps[idx].max() - age_min) / (age_max - age_min)
                # print(fmap_score)
                for key in age_indexes:
                    min_val = age_indexes[key][0]
                    max_val = age_indexes[key][1]
                    if idx >= min_val and idx < max_val:
                        age_scores[key] += (rank * fmap_score)

            for i, idx in enumerate(sorted_gender_indexes):
                rank = (i+1) / 4000
                fmap_score = (unit_gender_maps[idx].max() - gender_min) / (gender_max - gender_min)
                for key in gender_indexes:
                    min_val = gender_indexes[key][0]
                    max_val = gender_indexes[key][1]
                    if idx >= min_val and idx < max_val:
                        gender_scores[key] += (rank * fmap_score)

            for i, idx in enumerate(sorted_ethnic_indexes):
                rank = (i+1) / 4000
                fmap_score = (unit_ethnic_maps[idx].max() - ethnic_min) / (ethnic_max - ethnic_min)
                # print(unit_ethnic_maps[idx].max())
                for key in ethnic_indexes:
                    min_val = ethnic_indexes[key][0]
                    max_val = ethnic_indexes[key][1]
                    if idx >= min_val and idx < max_val:
                        ethnic_scores[key] += (rank * fmap_score)

            for i, idx in enumerate(sorted_skin_indexes):
                rank = (i+1) / 3000
                fmap_score = (unit_skin_maps[idx].max() - skin_min) / (skin_max - skin_min)
                # print(unit_ethnic_maps[idx].max())
                for key in skin_indexes:
                    min_val = skin_indexes[key][0]
                    max_val = skin_indexes[key][1]
                    if idx >= min_val and idx < max_val:
                        skin_scores[key] += (rank * fmap_score)

            age_scores['zero_to_twenty'] = age_scores['zero_to_twenty'] / 1000
            age_scores['twenty_to_forty'] = age_scores['twenty_to_forty'] / 1000
            age_scores['forty_to_sixty'] = age_scores['forty_to_sixty'] / 1000
            age_scores['sixty_plus'] = age_scores['sixty_plus'] / 1000

            ethnic_scores['white'] = ethnic_scores['white'] / 1000
            ethnic_scores['black'] = ethnic_scores['black'] / 1000
            ethnic_scores['asian'] = ethnic_scores['asian'] / 1000
            ethnic_scores['indian'] = ethnic_scores['indian'] / 1000

            gender_scores['male'] = gender_scores['male'] / 2000
            gender_scores['female'] = gender_scores['female'] / 2000

            skin_scores['bright'] = skin_scores['bright'] / 1500
            skin_scores['dark'] = skin_scores['dark'] / 1500

            age_probs = list(age_scores.values()) / sum(list(age_scores.values()))
            ethnic_probs = list(ethnic_scores.values()) / sum(list(ethnic_scores.values()))
            gender_probs = list(gender_scores.values()) / sum(list(gender_scores.values()))
            skin_probs = list(skin_scores.values()) / sum(list(skin_scores.values()))

            age_keys = list(age_scores.keys())
            ethnic_keys = list(ethnic_scores.keys())
            gender_keys = list(gender_scores.keys())
            skin_keys = list(skin_scores.keys())

            unit_prob_tally[ux+1]['age'] = dict((x, y) for x, y in zip(age_keys, age_probs))
            unit_prob_tally[ux+1]['gender'] = dict((x, y) for x, y in zip(gender_keys, gender_probs))
            unit_prob_tally[ux+1]['ethnic'] = dict((x, y) for x, y in zip(ethnic_keys, ethnic_probs))
            unit_prob_tally[ux+1]['skin'] = dict((x, y) for x, y in zip(skin_keys, skin_probs))

        # print(ethnic_scores)

        with open('{}/{}_NL_concept_probs.txt'.format(settings.NL_FOLDER, task), 'w') as txtfile:

            for unit in unit_prob_tally:

                curr_data = unit_prob_tally[unit]
                keys = list(curr_data.keys())
                # print(unit)
                txtfile.write('Unit : {}, spatially {}\n'.format(unit, unit_info[unit]))
                for j in keys:
                    txtfile.write('\t{}\n'.format(j.upper()))
                    txtfile.write('\t{\n')
                    for grp in curr_data[j]:
                        txtfile.write('\t\t {} : {}\n'.format(grp, curr_data[j][grp]))
                    txtfile.write('\t}\n')


# bias_analysis()
if __name__ == '__main__':

    dense_labels = ['AU_1', 'AU_2', 'AU_4', 'AU_5', 'AU_6', 'AU_9', 'AU_12', 'AU_15', 'AU_17', 'AU_20', 'AU_25', 'AU_26',
                    'nose', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'mouth', 'left_cheek', 'right_cheek',
                    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bushy_Eyebrows', 'No_Beard', 'Bags_Under_Eyes','Big_Lips',
                    'Big_Nose', 'Double_Chin', 'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Rosy_Cheeks',
                    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'Pointy_Nose', 'Smiling', 'Wearing_Lipstick']

    tasks = ['smile', 'age', 'gender', 'beauty', 'fairface', 'facenet']
    # units = [512, 512, 512, 512, 192, 512]
    sizes = [200, 128, 128, 224, 112, 224]

    for i, task in enumerate(tasks):
    # print(task, 'layer2')

    if task == 'age':
        from loader.model_loader_age import loadmodel
    if task == 'gender':
        from loader.model_loader_gender import loadmodel
    if task == 'beauty':
        from loader.model_loader_beauty import loadmodel
    if task == 'facenet':
        from loader.model_loader_face import loadmodel
    if task == 'fairface':
        from loader.model_loader_fairface import loadmodel
    if task == 'smile':
        from loader.model_loader_smile import loadmodel

    model = loadmodel(hook_feature)

    age_features(sizes[i], task, model)
    gender_features(sizes[i], task, model)
    ethnic_features(sizes[i], task, model)
    skin_features(sizes[i], task, model)

    bias_analysis()
