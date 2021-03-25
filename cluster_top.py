import torch
from torch import nn
import numpy as np
import settings
from matplotlib import pyplot as plt
import pickle
from tqdm import tqdm
from visual_dict_dataloader_new import visdict


def prob_dist(tasks, layer_name):


    data = visdict('visual_dictionary/', settings.BATCH_SIZE)
    dm = data.dense_label_mapping

    cluster_image_list = dict((x, []) for x in clusters)

    for region in clusters:
        concepts = clusters[region]
        for cpt in concepts:
            image_list = [x for x in dm[cpt] if not x in cluster_image_list[region]]
            for img_idx in image_list:
                cluster_image_list[region].append(img_idx)

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

        unit_ious = {}
        unit_subcats = {}

        for i in range(num_units):
            unit_scores = tally[i]
            subcat_idx = np.argsort(unit_scores)[-1]
            subcat = dense_labels[subcat_idx]

            if unit_scores[subcat_idx] < settings.SEG_THRESHOLD:
                continue

            unit_ious[i], unit_subcats[i] = unit_scores[subcat_idx], subcat

            for clus in clusters:
                if subcat in clusters[clus]:
                    cluster_info[i] = clus

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

            for cpt in concept_scores:

                dense_idx = dense_labels.index(cpt)
                cpt_iou = curr_scores[dense_idx]
                concept_scores[cpt] = concept_scores[cpt] * cpt_iou

            probs = list(concept_scores.values()) / sum(list(concept_scores.values()))
            # print(probs)
            keys = list(concept_scores.keys())

            unit_prob_tally[unit] = dict((x, y) for x, y in zip(keys, probs))
                # print(unit_prob_tally[unit][j])

        # print(unit_prob_tally)
        with open('{}/{}_{}_unit_concept_probs.pkl'.format(settings.OUTPUT_FOLDER, task, layer_name), 'wb') as handle:
            pickle.dump(unit_prob_tally, handle, protocol=pickle.HIGHEST_PROTOCOL)

def plot_hist(tasks, layer_name):

    tally_region = dict((x, dict((y, 0) for y in tasks)) for x in clusters)
    tally_type = dict((x, dict((y, 0) for y in tasks)) for x in types)

    tally_region['Unlocalizable'], tally_type['Unlocalizable'] = dict((y, 0) for y in tasks), dict((y, 0) for y in tasks)

    for task in tasks[:]:

        print(task)

        num_units = np.load(settings.OUTPUT_FOLDER + '/' + '{}_{}_tally20.npz'.format(task, layer_name))['tally'].shape[0]

        tally_og = dict((x.replace('_', ' '), 0) for x in dense_labels)

        with open('{}/{}_{}_unit_concept_probs.pkl'.format(settings.OUTPUT_FOLDER, task, layer_name), 'rb') as handle:
            data_og = pickle.load(handle)

        tally_region['Unlocalizable'][task] = num_units-len(data_og)
        tally_type['Unlocalizable'][task] = num_units-len(data_og)

        for unit in list(data_og.keys()):

            probs_og = data_og[unit]

            for og_concept in probs_og:

                if probs_og[og_concept] > (1.5/len(probs_og)):

                    tally_og[og_concept.replace('_', ' ')] += 1
                    for clus in clusters:
                        if og_concept in clusters[clus]:
                            tally_region[clus][task] += 1
                            break

                    for tp in types:
                        if og_concept in types[tp]:
                            tally_type[tp][task] += 1

        ind = np.arange(38)
        fig, ax = plt.subplots()

        fig.set_size_inches(14, 10)

        ax.bar(dense_labels, list(tally_og.values()), edgecolor='black')

        ax.tick_params(rotation=90, labelsize=7)

        ax.set_ylabel('Number of Interpretable Units')

        ax.set_title('{} Model, {} -- {Hierarchical Network Dissection}'.format(task.upper(), layer_name.upper()))

        plt.tight_layout()
        plt.savefig('plots/{}_{}_hist.png'.format(task.upper(), layer_name.upper()))
        # plt.show()
        plt.close()

    ind = np.arange(len(tasks))
    grp_width = 0.1
    fig, ax = plt.subplots()

    ax1 = ax.bar(ind - (5*grp_width/2), [tally_region['Eye_Region'][x] for x in tasks], width=grp_width, edgecolor='black')
    ax2 = ax.bar(ind - (3*grp_width/2), [tally_region['Cheek_Region'][x] for x in tasks], width=grp_width, edgecolor='black')
    ax3 = ax.bar(ind - (grp_width/2), [tally_region['Nose_Region'][x] for x in tasks], width=grp_width, edgecolor='black')
    ax4 = ax.bar(ind + (grp_width/2), [tally_region['Mouth_Region'][x] for x in tasks], width=grp_width, edgecolor='black')
    ax5 = ax.bar(ind + (3*grp_width/2), [tally_region['Chin_Region'][x] for x in tasks], width=grp_width, edgecolor='black')
    ax6 = ax.bar(ind + (5*grp_width/2), [tally_region['Unlocalizable'][x] for x in tasks], width=grp_width, edgecolor='black')

    plt.legend(('Eye_Region', 'Cheek_Region', 'Nose_Region', 'Mouth_Region', 'Chin_Region', 'Unlocalizable'), fontsize=8)
    plt.xticks(np.arange(6), ('Age', 'Gender', 'Beauty', 'Facenet', 'Fairface', 'Smile'))

    plt.tight_layout()
    plt.savefig('plots/compare/Region_comparison.png')
    # plt.show()
    plt.close()


    fig, ax = plt.subplots()

    ax1 = ax.bar(ind - (3*grp_width/2), [tally_type['AU'][x] for x in tasks], width=grp_width, edgecolor='black')
    ax2 = ax.bar(ind - (grp_width/2), [tally_type['Parts'][x] for x in tasks], width=grp_width, edgecolor='black')
    ax3 = ax.bar(ind + (grp_width/2), [tally_type['Attr'][x] for x in tasks], width=grp_width, edgecolor='black')
    ax4 = ax.bar(ind + (3*grp_width/2), [tally_type['Unlocalizable'][x] for x in tasks], width=grp_width, edgecolor='black')

    plt.legend(('AU', 'Parts', 'Attr', 'Unlocalizable'), fontsize=8)
    plt.xticks(np.arange(6), ('Age', 'Gender', 'Beauty', 'Facenet', 'Fairface', 'Smile'))

    plt.tight_layout()
    plt.savefig('plots/compare/Concept_type_comparison.png')
    # plt.show()
    plt.close()





if __name__ == '__main__':

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

    types = {'AU':['AU_1', 'AU_2', 'AU_4', 'AU_5', 'AU_6', 'AU_9', 'AU_12', 'AU_15', 'AU_17', 'AU_20', 'AU_25', 'AU_26'],
             'Parts':['nose', 'left_brow', 'right_brow', 'left_eye', 'right_eye', 'mouth', 'left_cheek', 'right_cheek'],
             'Attr':['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bushy_Eyebrows', 'No_Beard', 'Bags_Under_Eyes','Big_Lips',
                    'Big_Nose', 'Double_Chin', 'Eyeglasses', 'Goatee', 'High_Cheekbones', 'Rosy_Cheeks',
                    'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'Pointy_Nose', 'Smiling', 'Wearing_Lipstick']}

    tasks = ['age', 'gender', 'beauty', 'facenet', 'fairface', 'smile']

    # plot_cluster_figure()
    prob_dist(tasks, layer_name)
    plot_hist(tasks, layer_name)
