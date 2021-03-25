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

def age_features(size, curr_task, curr_model, layer_name):

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

    np.savez('{}/{}_model_bias(age)_{}.npz'.format(settings.NL_FOLDER, curr_task, layer_name), zero=feature_dict['zero_to_twenty'],
                                                                                               twenty=feature_dict['twenty_to_forty'],
                                                                                               forty=feature_dict['forty_to_sixty'],
                                                                                               sixty=feature_dict['sixty_plus'])


def gender_features(size, curr_task, curr_model, layer_name):

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

    np.savez('{}/{}_model_bias(gender)_{}.npz'.format(settings.NL_FOLDER, curr_task, layer_name),
                                               male=feature_dict['Male'],
                                               female=feature_dict['Female'])


def ethnic_features(size, curr_task, curr_model, layer_name):

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
                first = False

            features[idx:idx+1] = curr_activation

        feature_dict[grp] = features

    np.savez('{}/{}_model_bias(ethnic)_{}.npz'.format(settings.NL_FOLDER, curr_task, layer_name), white=feature_dict['white'],
                                                                                                  black=feature_dict['black'],
                                                                                                  asian=feature_dict['asian'],
                                                                                                  indian=feature_dict['indian'])

def skin_features(size, curr_task, curr_model, layer_name):

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

    np.savez('{}/{}_model_bias(skin_tone)_{}.npz'.format(settings.NL_FOLDER, curr_task, layer_name), bright=feature_dict['bright'], dark=feature_dict['dark'])


def bias_analysis(plot=False, tasks, layer_name):

    for task in tqdm(tasks):
        # print(task)

        if plot:
            zeros = []
            twentys = []
            fortys = []
            sixtys = []

            males = []
            females = []

            whites = []
            asians = []
            blacks = []
            indians = []

            brights = []
            darks = []

        age_features = np.load('{}/{}_model_bias(age)_{}.npz'.format(settings.NL_FOLDER, task, layer_name))
        gender_features = np.load('{}/{}_model_bias(gender)_{}.npz'.format(settings.NL_FOLDER, task, layer_name))
        ethnic_features = np.load('{}/{}_model_bias(ethnic)_{}.npz'.format(settings.NL_FOLDER, task, layer_name))
        skin_features = np.load('{}/{}_model_bias(skin_tone)_{}.npz'.format(settings.NL_FOLDER, task, layer_name))
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

        # print(all_age_maps.shape, all_gender_maps.shape, all_ethnic_maps.shape, all_skin_maps.shape)

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

            age_flag = True
            gender_flag = True
            ethnic_flag = True
            skin_flag = True

            unit_prob_tally[ux] = {'age':{}, 'gender':{}, 'ethnic':{}, 'skin':{}}
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

            if (age_max - age_min) == 0:

                unit_prob_tally[ux]['age'] = {
                                                'zero_to_twenty':0,
                                                'twenty_to_forty':0,
                                                'forty_to_sixty':0,
                                                'sixty_plus':0
                                             }
                zeros.append(0)
                twentys.append(0)
                fortys.append(0)
                sixtys.append(0)
                age_flag = False

            if (gender_max - gender_min) == 0:

                unit_prob_tally[ux]['gender'] = {
                                                   'male':0,
                                                   'female':0,
                                                }
                males.append(0)
                females.append(0)
                male_flag = False

            if (ethnic_max - ethnic_min) == 0:

                unit_prob_tally[ux]['ethnic'] = {
                                                    'white':0,
                                                    'black':0,
                                                    'asian':0,
                                                    'indian':0
                                                }
                whites.append(0)
                asians.append(0)
                blacks.append(0)
                indians.append(0)
                ethnic_flag = False

            if (skin_max - skin_min) == 0:

                unit_prob_tally[ux]['skin'] = {
                                                'bright':0,
                                                'dark':0,
                                              }
                males.append(0)
                females.append(0)
                skin_flag = False

            if age_flag:

                sorted_age_indexes = np.argsort(np.max(unit_age_maps.reshape(4000, -1), axis=-1))

                for i, idx in enumerate(sorted_age_indexes):
                    rank = (i+1) / 4000
                    fmap_score = (unit_age_maps[idx].max() - age_min) / (age_max - age_min)
                    # print(fmap_score)
                    for key in age_indexes:
                        min_val = age_indexes[key][0]
                        max_val = age_indexes[key][1]
                        if idx >= min_val and idx < max_val:
                            age_scores[key] += (rank * fmap_score)

                age_scores['zero_to_twenty'] = age_scores['zero_to_twenty'] / 1000
                age_scores['twenty_to_forty'] = age_scores['twenty_to_forty'] / 1000
                age_scores['forty_to_sixty'] = age_scores['forty_to_sixty'] / 1000
                age_scores['sixty_plus'] = age_scores['sixty_plus'] / 1000

                age_probs = list(age_scores.values()) / sum(list(age_scores.values()))
                age_keys = list(age_scores.keys())

                unit_prob_tally[ux]['age'] = dict((x, y) for x, y in zip(age_keys, age_probs))

                if plot:
                    zeros.append(age_probs[0])
                    twentys.append(age_probs[1])
                    fortys.append(age_probs[2])
                    sixtys.append(age_probs[3])

            if gender_flag:

                sorted_gender_indexes = np.argsort(np.max(unit_gender_maps.reshape(4000, -1), axis=-1))\

                for i, idx in enumerate(sorted_gender_indexes):
                    rank = (i+1) / 4000
                    fmap_score = (unit_gender_maps[idx].max() - gender_min) / (gender_max - gender_min)
                    for key in gender_indexes:
                        min_val = gender_indexes[key][0]
                        max_val = gender_indexes[key][1]
                        if idx >= min_val and idx < max_val:
                            gender_scores[key] += (rank * fmap_score)

                gender_scores['male'] = gender_scores['male'] / 2000
                gender_scores['female'] = gender_scores['female'] / 2000

                gender_probs = list(gender_scores.values()) / sum(list(gender_scores.values()))
                gender_keys = list(gender_scores.keys())
                unit_prob_tally[ux]['gender'] = dict((x, y) for x, y in zip(gender_keys, gender_probs))

                if plot:
                    males.append(gender_probs[0])
                    females.append(gender_probs[1])

            if ethnic_flag:

                sorted_ethnic_indexes = np.argsort(np.max(unit_ethnic_maps.reshape(4000, -1), axis=-1))

                for i, idx in enumerate(sorted_ethnic_indexes):
                    rank = (i+1) / 4000
                    fmap_score = (unit_ethnic_maps[idx].max() - ethnic_min) / (ethnic_max - ethnic_min)
                    # print(unit_ethnic_maps[idx].max())
                    for key in ethnic_indexes:
                        min_val = ethnic_indexes[key][0]
                        max_val = ethnic_indexes[key][1]
                        if idx >= min_val and idx < max_val:
                            ethnic_scores[key] += (rank * fmap_score)

                ethnic_scores['white'] = ethnic_scores['white'] / 1000
                ethnic_scores['black'] = ethnic_scores['black'] / 1000
                ethnic_scores['asian'] = ethnic_scores['asian'] / 1000
                ethnic_scores['indian'] = ethnic_scores['indian'] / 1000

                ethnic_probs = list(ethnic_scores.values()) / sum(list(ethnic_scores.values()))
                ethnic_keys = list(ethnic_scores.keys())
                unit_prob_tally[ux]['ethnic'] = dict((x, y) for x, y in zip(ethnic_keys, ethnic_probs))

                if plot:
                    whites.append(ethnic_probs[0])
                    blacks.append(ethnic_probs[1])
                    asians.append(ethnic_probs[2])
                    indians.append(ethnic_probs[3])

            if skin_flag:

                sorted_skin_indexes = np.argsort(np.max(unit_skin_maps.reshape(3000, -1), axis=-1))

                for i, idx in enumerate(sorted_skin_indexes):
                    rank = (i+1) / 3000
                    fmap_score = (unit_skin_maps[idx].max() - skin_min) / (skin_max - skin_min)
                    # print(unit_ethnic_maps[idx].max())
                    for key in skin_indexes:
                        min_val = skin_indexes[key][0]
                        max_val = skin_indexes[key][1]
                        if idx >= min_val and idx < max_val:
                            skin_scores[key] += (rank * fmap_score)

                skin_scores['bright'] = skin_scores['bright'] / 1500
                skin_scores['dark'] = skin_scores['dark'] / 1500

                skin_probs = list(skin_scores.values()) / sum(list(skin_scores.values()))
                skin_keys = list(skin_scores.keys())
                unit_prob_tally[ux]['skin'] = dict((x, y) for x, y in zip(skin_keys, skin_probs))

                if plot:
                    brights.append(skin_probs[0])
                    darks.append(skin_probs[1])

        # print(ethnic_scores)

        with open('{}/{}_{}_nl_probs.pkl'.format(settings.NL_FOLDER, task, layer_name), 'wb') as handle:

            pickle.dump(unit_prob_tally, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open('{}/{}_{}_nl_probs.txt'.format(settings.NL_FOLDER, task, layer_name), 'w') as txtfile:

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

        if plot:

            if not os.path.exists('plots/probs'):
                os.makedirs('plots/probs')

            descending_age_index = np.argsort(zeros)[::-1]
            descending_gender_index = np.argsort(males)[::-1]
            descending_ethnic_index = np.argsort(whites)[::-1]
            descending_skin_index = np.argsort(brights)[::-1]

            zeros = [zeros[x] for x in descending_age_index]
            twentys = [twentys[x] for x in descending_age_index]
            fortys = [fortys[x] for x in descending_age_index]
            sixtys = [sixtys[x] for x in descending_age_index]

            males = [males[x] for x in descending_gender_index]
            females = [females[x] for x in descending_gender_index]

            whites = [whites[x] for x in descending_ethnic_index]
            blacks = [blacks[x] for x in descending_ethnic_index]
            asians = [asians[x] for x in descending_ethnic_index]
            indians = [indians[x] for x in descending_ethnic_index]

            brights = [brights[x] for x in descending_skin_index]
            darks = [darks[x] for x in descending_skin_index]

            index = np.arange(num_units)

            fig, ax = plt.subplots()
            fig.set_size_inches(14,10)

            ax.bar(index, zeros) #, edgecolor='black')
            ax.bar(index, twentys, bottom=zeros) #, edgecolor='black')
            ax.bar(index, fortys, bottom=[(x+y) for x,y in zip(zeros,twentys)])
            ax.bar(index, sixtys, bottom=[(x+y+z) for x,y,z in zip(zeros,twentys,fortys)])
            ax.set_xlabel("Units")
            ax.set_ylabel("Probability")

            # plt.tight_layout()
            plt.title('{} Model - {},\
                       Age Group Probs\n 0-20:Avg Prob - {:.4f}, 20-40:Avg Prob - {:.4f}, 40-60:Avg Prob - {:.4f}, 60+:Avg Prob - {:.4f}\
                       \n 0-20:Std Dev - {:.4f},431 , 20-40:Std Dev - {:.4f}, 40-60:Std Dev - {:.4f}, 60+:Std Dev - {:.4f}'.format(task.upper(),
                                                                                                                                    layer_name,
                                                                                                                                    mean(zeros),
                                                                                                                                    mean(twentys),
                                                                                                                                    mean(fortys),
                                                                                                                                    mean(sixtys),
                                                                                                                                    stdev(zeros),
                                                                                                                                    stdev(twentys),
                                                                                                                                    stdev(fortys),
                                                                                                                                    stdev(sixtys)))

            plt.legend(('0-20', '20-40', '40-60', '60+'), bbox_to_anchor=(1, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('plots/probs/{}_{}_probs(age).png'.format(task.upper(), layer_name))
            plt.close()

            fig, ax = plt.subplots()

            ax.bar(index, males) #, edgecolor='black')
            ax.bar(index, females, bottom=males) #, edgecolor='black')
            ax.set_xlabel("Units")
            ax.set_ylabel("Probability")

            # plt.tight_layout()
            plt.title('{} Model - {}, Gender Probs\n Male Avg Prob - {:.4f}, Female Avg Prob - {:.4f}\n Std Dev - {:.4f}'.format(task.upper(),
                                                                                                                                 layer_name,
                                                                                                                                 mean(males),
                                                                                                                                 mean(females),
                                                                                                                                 stdev(males)))

            plt.legend(('Male', 'Female'), bbox_to_anchor=(1, 1), loc='upper left')
            plt.tight_layout()
            # plt.show()
            plt.savefig('plots/probs/{}_{}_probs(gender).png'.format(task.upper(), layer_name))
            plt.close()

            fig, ax = plt.subplots()
            fig.set_size_inches(14,10)

            ax.bar(index, whites) #, edgecolor='black')
            ax.bar(index, blacks, bottom=whites) #, edgecolor='black')
            ax.bar(index, asians, bottom=[(x+y) for x,y in zip(whites,blacks)])
            ax.bar(index, indians, bottom=[(x+y+z) for x,y,z in zip(whites,blacks,asians)])
            ax.set_xlabel("Units")
            ax.set_ylabel("Probability")

            # plt.tight_layout()
            plt.title('{} Model - {},\
                       Ethnic Probs\n Whites Avg Prob - {:.4f}, Blacks Avg Prob - {:.4f}, Asians Avg Prob - {:.4f}, Indians Avg Prob - {:.4f}\
                       \n Whites Std Dev - {:.4f},431 , Blacks Std Dev - {:.4f}, Asians Std Dev - {:.4f}, Indians Std Dev - {:.4f}'.format(task.upper(),
                                                                                                                                           layer_name,
                                                                                                                                           mean(whites),
                                                                                                                                           mean(blacks),
                                                                                                                                           mean(asians),
                                                                                                                                           mean(indians),
                                                                                                                                           stdev(whites),
                                                                                                                                           stdev(blacks),
                                                                                                                                           stdev(asians),
                                                                                                                                           stdev(indians)))

            plt.legend(('Whites', 'Blacks', 'Asians', 'Indians'), bbox_to_anchor=(1, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('plots/probs/{}_{}_probs(ethnic).png'.format(task.upper(), layer_name))
            plt.close()

            fig, ax = plt.subplots()

            ax.bar(index, brights) #, edgecolor='black')
            ax.bar(index, darks, bottom=brights) #, edgecolor='black')
            ax.set_xlabel("Units")
            ax.set_ylabel("Probability")

            # plt.tight_layout()
            plt.title('{} Model - {}, Skin Tone Probs\n Bright Avg Prob - {:.4f}, Dark Avg Prob - {:.4f}\n Std Dev - {:.4f}'.format(task.upper(),
                                                                                                                                    layer_name,
                                                                                                                                    mean(brights),
                                                                                                                                    mean(darks),
                                                                                                                                    stdev(darks)))

            plt.legend(('Bright', 'Dark'), bbox_to_anchor=(1, 1), loc='upper left')
            plt.tight_layout()
            # plt.show()
            plt.savefig('plots/probs/{}_{}_probs(skin_tone).png'.format(task.upper(), layer_name))
            plt.close()

def comparison_plots(tasks, layer_name):

    age_biased_tally = dict((x, {'zero_to_twenty':0, 'twenty_to_forty':0, 'forty_to_sixty':0, 'sixty_plus':0, 'un':0}) for x in tasks)
    gender_biased_tally = dict((x, {'male':0, 'female':0, 'un':0}) for x in tasks)
    ethnic_biased_tally = dict((x, {'white':0, 'black':0, 'asian':0, 'indian':0, 'un':0}) for x in tasks)
    skin_biased_tally = dict((x, {'bright':0, 'dark':0, 'un':0}) for x in tasks)


    for task in tasks:

        print(task)

        with open('{}/{}_{}_nl_probs.pkl'.format(settings.NL_FOLDER, task, layer_name), 'rb') as handle:
            data = pickle.load(handle)

        for unit in data:

            age_dict = data[unit]['age']
            gender_dict = data[unit]['gender']
            ethnic_dict = data[unit]['ethnic']
            skin_dict = data[unit]['skin']

            age_bias_flag = False
            gender_bias_flag = False
            ethnic_bias_flag = False
            skin_bias_flag = False

            for sub_cat in age_dict:

                if age_dict[sub_cat] >= 0.3:

                    age_biased_tally[task][sub_cat] += 1
                    age_bias_flag = True

            if not age_bias_flag:
                age_biased_tally[task]['un'] += 1

            for sub_cat in gender_dict:

                if gender_dict[sub_cat] >= 0.55:

                    gender_biased_tally[task][sub_cat] += 1
                    gender_bias_flag = True

            if not gender_bias_flag:
                gender_biased_tally[task]['un'] += 1

            for sub_cat in ethnic_dict:

                if ethnic_dict[sub_cat] >= 0.3:

                    ethnic_biased_tally[task][sub_cat] += 1
                    ethnic_bias_flag = True

            if not ethnic_bias_flag:
                ethnic_biased_tally[task]['un'] += 1

            for sub_cat in skin_dict:

                if skin_dict[sub_cat] >= 0.55:

                    skin_biased_tally[task][sub_cat] += 1
                    skin_bias_flag = True

            if not skin_bias_flag:
                skin_biased_tally[task]['un'] += 1

    if not os.path.exists('plots/compare'):
        os.makedirs('plots/compare')

    ind = np.arange(len(tasks))
    grp_width = 0.1
    fig, ax = plt.subplots()

    ax1 = ax.bar(ind - (2*grp_width), [age_biased_tally[x]['zero_to_twenty'] for x in tasks], width=grp_width, edgecolor='black')
    ax2 = ax.bar(ind - grp_width, [age_biased_tally[x]['twenty_to_forty'] for x in tasks], width=grp_width, edgecolor='black')
    ax3 = ax.bar(ind, [age_biased_tally[x]['forty_to_sixty'] for x in tasks], width=grp_width, edgecolor='black')
    ax4 = ax.bar(ind + grp_width, [age_biased_tally[x]['sixty_plus'] for x in tasks], width=grp_width, edgecolor='black')
    ax5 = ax.bar(ind + (2*grp_width), [age_biased_tally[x]['un'] for x in tasks], width=grp_width, edgecolor='black')

    plt.legend(('0 - 20', '20 - 40', '40 - 60', '60+', 'Unbiased'), fontsize=8)
    plt.xticks(np.arange(6), ('Age', 'Gender', 'Beauty', 'Facenet', 'Fairface', 'Smile'))
    # plt.yticks(np.arange(0, 500, 100))
    plt.tight_layout()
    plt.savefig('plots/compare/Age_groups_comparison.png')
    # plt.show()
    plt.close()

    fig, ax = plt.subplots()

    ax1 = ax.bar(ind - (2*grp_width), [ethnic_biased_tally[x]['white'] for x in tasks], width=grp_width, edgecolor='black')
    ax2 = ax.bar(ind - grp_width, [ethnic_biased_tally[x]['black'] for x in tasks], width=grp_width, edgecolor='black')
    ax3 = ax.bar(ind, [ethnic_biased_tally[x]['asian'] for x in tasks], width=grp_width, edgecolor='black')
    ax4 = ax.bar(ind + grp_width, [ethnic_biased_tally[x]['indian'] for x in tasks], width=grp_width, edgecolor='black')
    ax5 = ax.bar(ind + (2*grp_width), [ethnic_biased_tally[x]['un'] for x in tasks], width=grp_width, edgecolor='black')

    plt.legend(('White', 'Black', 'Asian', 'Indian', 'Unbiased'), fontsize=8)
    plt.xticks(np.arange(6), ('Age', 'Gender', 'Beauty', 'Facenet', 'Fairface', 'Smile'))
    # plt.yticks(np.arange(0, 500, 100))
    plt.tight_layout()
    plt.savefig('plots/compare/Ethnic_groups_comparison.png')
    # plt.show()
    plt.close()

    fig, ax = plt.subplots()

    ax1 = ax.bar(ind - grp_width, [skin_biased_tally[x]['bright'] for x in tasks], width=grp_width, edgecolor='black')
    ax2 = ax.bar(ind, [skin_biased_tally[x]['dark'] for x in tasks], width=grp_width, edgecolor='black')
    ax3 = ax.bar(ind + grp_width, [skin_biased_tally[x]['un'] for x in tasks], width=grp_width, edgecolor='black')

    plt.legend(('Bright', 'Dark', 'Unbiased'), fontsize=8)
    plt.xticks(np.arange(6), ('Age', 'Gender', 'Beauty', 'Facenet', 'Fairface', 'Smile'))
    # plt.yticks(np.arange(0, 500, 100))
    plt.tight_layout()
    plt.savefig('plots/compare/Skin_groups_comparison.png')
    # plt.show()
    plt.close()

    fig, ax = plt.subplots()

    ax1 = ax.bar(ind - grp_width, [gender_biased_tally[x]['male'] for x in tasks], width=grp_width, edgecolor='black')
    ax2 = ax.bar(ind, [gender_biased_tally[x]['female'] for x in tasks], width=grp_width, edgecolor='black')
    ax3 = ax.bar(ind + grp_width, [gender_biased_tally[x]['un'] for x in tasks], width=grp_width, edgecolor='black')

    plt.legend(('Male', 'Female', 'Unbiased'), fontsize=8)
    plt.xticks(np.arange(6), ('Age', 'Gender', 'Beauty', 'Facenet', 'Fairface', 'Smile'))
    # plt.yticks(np.arange(0, 500, 100))
    plt.tight_layout()
    plt.savefig('plots/compare/Gender_groups_comparison.png')
    # plt.show()
    plt.close()


if __name__ == '__main__':

    dense_labels = ['AU_1', 'AU_2', 'AU_4', 'AU_5', 'AU_6', 'AU_9', 'AU_12', 'AU_15', 'AU_17', 'AU_20', 'AU_25', 'AU_26',
                    'nose', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'mouth', 'left_cheek', 'right_cheek',
                    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bushy_Eyebrows', 'No_Beard', 'Bags_Under_Eyes','Big_Lips',
                    'Big_Nose', 'Double_Chin', 'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Rosy_Cheeks',
                    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'Pointy_Nose', 'Smiling', 'Wearing_Lipstick']

    tasks = ['smile', 'age', 'gender', 'beauty', 'fairface', 'facenet']
    # units = [512, 512, 512, 512, 192, 512]
    sizes = [200, 128, 128, 224, 112, 224]
    layer_name = settings.LAYER_NAME

    generate = True # Set False if feature.npz files have already been generated

    if generate:
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

            age_features(sizes[i], task, model, layer_name)
            gender_features(sizes[i], task, model, layer_name)
            ethnic_features(sizes[i], task, model, layer_name)
            skin_features(sizes[i], task, model, layer_name)

    else:
        pass

    bias_analysis(tasks, layer_name)
    comparison_plots(tasks, layer_name)
