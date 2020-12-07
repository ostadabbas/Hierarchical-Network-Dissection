import torch
from torch import nn
import numpy as np
import settings
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
from visual_dict_dataloader_new import visdict

clusters = {'Eye_Region': ['AU_1', 'AU_2', 'AU_4', 'AU_5', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'Arched_Eyebrows', 'Bushy_Eyebrows', 'Eyeglasses', 'Narrow_Eyes'],
            'Cheek_Region': ['AU_6', 'left_cheek', 'right_cheek', 'Bags_Under_Eyes', 'High_Cheekbones', 'Rosy_Cheeks', 'No_Beard', '5_o_Clock_Shadow'],
            'Nose_Region': ['AU_9', 'nose', 'Big_Nose', 'Pointy_Nose'],
            'Mouth_Region': ['AU_12', 'AU_15', 'AU_20', 'AU_25', 'mouth', 'Big_Lips', 'Mouth_Slightly_Open', 'Mustache', 'Smiling', 'Wearing_Lipstick'],
            'Chin_Region': ['AU_17', 'AU_26', 'Double_Chin', 'Goatee']}

dense_labels = ['AU_1', 'AU_2', 'AU_4', 'AU_5', 'AU_6', 'AU_9', 'AU_12', 'AU_15', 'AU_17', 'AU_20', 'AU_25', 'AU_26',
                'nose', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'mouth', 'left_cheek', 'right_cheek',
                '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bushy_Eyebrows', 'No_Beard', 'Bags_Under_Eyes','Big_Lips',
                'Big_Nose', 'Double_Chin', 'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Rosy_Cheeks',
                'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'Pointy_Nose', 'Smiling', 'Wearing_Lipstick']

tasks = ['age', 'gender', 'beauty', 'facenet', 'fairface', 'smile']
# tasks = ['age', 'fairface']
region_tally = dict((x, {'age':0, 'gender':0, 'beauty':0, 'facenet':0, 'fairface':0, 'smile':0}) for x in clusters)

for task in tasks:
    tally_file = np.load(settings.OUTPUT_FOLDER + '/' + '{}_layer4_tally20.npz'.format(task))
    scores = tally_file['tally']
    num_units = scores.shape[0]

    for i in range(num_units):

        sub_cat_idx = np.nanargmax(scores[i])
        sub_cat = dense_labels[sub_cat_idx]
        if scores[i][sub_cat_idx] < 0.04:
            continue

        for region in clusters:
            if sub_cat in clusters[region]:
                region_tally[region][task] += 1
                break

def plot_cluster_figure():
    fig, ax = plt.subplots()

    first_value = list(region_tally['Eye_Region'].values())
    second_value = [(a+b) for a, b in zip(list(region_tally['Eye_Region'].values()), list(region_tally['Cheek_Region'].values()))]
    third_value = [(a+b+c) for a, b, c in zip(list(region_tally['Eye_Region'].values()), list(region_tally['Cheek_Region'].values()), list(region_tally['Nose_Region'].values()))]
    fourth_value = [(a+b+c+d) for a, b, c, d in zip(list(region_tally['Eye_Region'].values()), list(region_tally['Cheek_Region'].values()), list(region_tally['Nose_Region'].values()), list(region_tally['Mouth_Region'].values()))]
    fifth_value = [(a+b+c+d+e) for a, b, c, d, e in zip(list(region_tally['Eye_Region'].values()), list(region_tally['Cheek_Region'].values()), list(region_tally['Nose_Region'].values()), list(region_tally['Mouth_Region'].values()), list(region_tally['Chin_Region'].values()))]
    empty = [512-(a+b+c+d+e) for a, b, c, d, e in zip(list(region_tally['Eye_Region'].values()), list(region_tally['Cheek_Region'].values()), list(region_tally['Nose_Region'].values()), list(region_tally['Mouth_Region'].values()), list(region_tally['Chin_Region'].values()))]
    empty[3] = empty[3] - 320

    ax1 = ax.bar(np.arange(5), list(region_tally['Eye_Region'].values()), width=0.2)
    ax2 = ax.bar(np.arange(5), list(region_tally['Cheek_Region'].values()), bottom=first_value, width=0.2)
    ax3 = ax.bar(np.arange(5), list(region_tally['Nose_Region'].values()), bottom=second_value, width=0.2)
    ax4 = ax.bar(np.arange(5), list(region_tally['Mouth_Region'].values()), bottom=third_value, width=0.2)
    ax5 = ax.bar(np.arange(5), list(region_tally['Chin_Region'].values()), bottom=fourth_value, width=0.2)
    ax6 = ax.bar(np.arange(5), empty, bottom=fifth_value, width=0.2, color='gray')
    plt.legend(('Eye Region', 'Cheek Region', 'Nose Region', 'Mouth Region', 'Chin Region', 'Uniterpretable'),
                bbox_to_anchor=(0.85,0.92), loc="upper right", bbox_transform=plt.gcf().transFigure)
    plt.xticks(np.arange(5), ('Age', 'Gender', 'Beauty', 'Facenet', 'Fairface'))
    plt.yticks(np.arange(0, 700, 50))
    plt.tight_layout()

    plt.savefig('Region_Comparison.png')
    # plt.show()


def prob_dist():

    tasks = ['age', 'gender', 'beauty', 'facenet', 'fairface', 'smile']
    layer_name = 'layer4'
    data = visdict('visual_dictionary/', settings.BATCH_SIZE)
    dm = data.dense_label_mapping

    for task in tasks:

        # print(task, '\n')

        tally_file = np.load(settings.OUTPUT_FOLDER + '/' + '{}_{}_tally20.npz'.format(task, layer_name))
        tally = tally_file['tally']
        # num_units = tally.shape[0]
        cluster_info = {}
        unit_prob_tally = {}
        size = np.load(settings.OUTPUT_FOLDER + '/' + '{}_{}_feature_size.npy'.format(task, layer_name))[0]
        features = np.memmap(settings.OUTPUT_FOLDER + '/' + '{}_{}.mmap'.format(task, layer_name), dtype=float, mode='r', shape=tuple(size))

        num_units = features.shape[1]
        num_images = features.shape[0]

        cluster_image_list = dict((x, []) for x in clusters)

        for region in clusters:
            concepts = clusters[region]
            for cpt in concepts:
                image_list = [x for x in dm[cpt] if not x in cluster_image_list[region]]
                for img_idx in image_list:
                    cluster_image_list[region].append(img_idx)

        unit_ious = {}
        unit_subcats = {}

        for i in range(num_units):
            unit_scores = tally[i]
            subcat_idx = np.argsort(unit_scores)[-1]
            subcat = dense_labels[subcat_idx]
            # subcat2 = dense_labels[subcat_idx2]
            # subcat3 = dense_labels[subcat_idx3]

            if unit_scores[subcat_idx] < settings.SEG_THRESHOLD:
                continue

            unit_ious[i], unit_subcats[i] = unit_scores[subcat_idx], subcat

            # print(i, unit_scores[subcat_idx1], unit_scores[subcat_idx2], unit_scores[subcat_idx3])

            for clus in clusters:
                if subcat in clusters[clus]:
                    cluster_info[i] = clus

    # with open('{}/age_layer4_all_maps.pickle'.format(settings.OUTPUT_FOLDER), 'rb') as file:
    #     maps = pickle.load(file)

    # with open('{}/age_layer4_all_scores.pickle'.format(settings.OUTPUT_FOLDER), 'rb') as file:
    #     scores = pickle.load(file)
    # print(cluster_info)

        count = 0

        for unit in tqdm(cluster_info):
            # if count == 3:
            #     break
            unit_prob_tally[unit]
            # count += 1

            curr_cluster = cluster_info[unit]
            cluster_concepts = clusters[curr_cluster]
            # print(unit)
            concept_scores = dict((x, 0) for x in cluster_concepts)
            unit_maps = features[:, unit, :, :]
            range_min = unit_maps.min()
            range_max = unit_maps.max()
            h, w = unit_maps.shape[1], unit_maps.shape[2]
            num_cluster_images = len(cluster_image_list[curr_cluster])
            selected_maps = np.zeros((num_cluster_images, h, w))

            curr_image_list = cluster_image_list[curr_cluster]
            image_mapping = {}

            for i, img_idx in enumerate(curr_image_list):
                selected_maps[i] = unit_maps[img_idx]
                image_mapping[i] = img_idx

            sorted_index = np.argsort(np.max(selected_maps.reshape(num_cluster_images, -1), axis=1))
            # print(len(sorted_index))
            # print(sorted_index.shape)
            # print(dm['left_cheek'], dm['right_cheek'])
            # print(cluster_image_list['Cheek_Region'])

            for i, idx in enumerate(sorted_index):
                rank = (i+1) / num_cluster_images
                fmap_score = (selected_maps[idx].max() - range_min) / (range_max - range_min)
                for cpt in cluster_concepts:
                    if image_mapping[idx] in dm[cpt]:
                        concept_scores[cpt] += (rank * fmap_score)

            for cpt in cluster_concepts:
                concept_scores[cpt] = concept_scores[cpt] / len(dm[cpt])

            # print(concept_scores)

            # probs = softmax(list(concept_scores.values()))
            probs = list(concept_scores.values()) / sum(list(concept_scores.values()))
            # print(probs)
            keys = list(concept_scores.keys())

            unit_prob_tally[unit] = dict((x, y) for x, y in zip(keys, probs))
                # print(unit_prob_tally[unit][j])

        # print(unit_prob_tally)
        with open('{}/{}_top_concept_probs.txt'.format(settings.OUTPUT_FOLDER, task), 'w') as txtfile:
            for unit in unit_prob_tally:
                # print(unit)
                txtfile.write('Unit : {}\n'.format(unit))
                # for j in range(1, 4):
                txtfile.write('\t Concept: {}, IoU : {:.4f}, Cluster : {}\n'.format(unit_subcats[unit],
                                                                                    unit_ious[unit],
                                                                                    cluster_info[unit]))

                txtfile.write('\t{\n')
                for cpt in unit_prob_tally[unit]:
                    txtfile.write('\t\t {} : {}\n'.format(cpt, unit_prob_tally[unit][cpt]))
                txtfile.write('\t}\n')

if __name__ == '__main__':

    # plot_cluster_figure()
    # prob_dist()
