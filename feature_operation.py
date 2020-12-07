import os
import numpy as np
import torch
import settings
import time
import vecquantile as vecquantile
from visual_dict_dataloader_new import visdict
import cv2
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image
import concurrent.futures
import pickle


features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


class FeatureOperator:

    def __init__(self):
        if not os.path.exists(settings.OUTPUT_FOLDER):
            os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'image'))
        # self.image_data = visdict('visual_dictionary/visual_dict.csv')
        self.loader = visdict('visual_dictionary/', settings.BATCH_SIZE)
        self.tally_loader = visdict('visual_dictionary/', settings.TALLY_BATCH_SIZE)

    def feature_extraction(self, model=None, memmap=True, face_crop=False):
        loader = self.loader
        # extract the max value activaiton for each image
        wholefeatures = [None] * len(settings.FEATURE_NAMES)
        features_size = [None] * len(settings.FEATURE_NAMES)
        features_size_file = os.path.join(settings.OUTPUT_FOLDER, "{}_feature_size.npy".format(settings.FEATURE_NAMES[0]))

        if memmap:
            skip = True
            mmap_files =  [os.path.join(settings.OUTPUT_FOLDER, "%s.mmap" % feature_name)  for feature_name in  settings.FEATURE_NAMES]
            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file)
            else:
                skip = False
            for i, mmap_file in enumerate(mmap_files):
                if os.path.exists(mmap_file) and features_size[i] is not None:
                    print('loading features %s' % settings.FEATURE_NAMES[i], features_size[i])
                    wholefeatures[i] = np.memmap(mmap_file, dtype=float, mode='r', shape=tuple(features_size[i]))
                else:
                    print('file missing, loading from scratch')
                    skip = False
            if skip:
                return wholefeatures, features_size

        gen = self.loader.get_batches(only_image=True, face_crop=face_crop)
        num_batches = self.loader.num_batches

        for batch_idx in range(num_batches):
            del features_blobs[:]
            batch = next(gen)
            # model_inp = torch.cat([x['image'].unsqueeze(0) for x in batch])
            # print(inp.shape)
            print('extracting feature from batch %d / %d' % (batch_idx+1, num_batches))
            model_inp = torch.cat([x.unsqueeze(0) for x in batch])
            # print(model_inp.shape)
            # print(features_blobs[0].shape)

            if settings.GPU:
                model_inp = model_inp.cuda()
                model = model.cuda()

            logit = model.forward(model_inp)
            # print(features_blobs[0].shape)
            '''
            while np.isnan(logit.data.cpu().max()):
                print("nan")
                del features_blobs[:]
                logit = model.forward(model_inp)
            '''

            if len(features_blobs[0].shape) == 4 and wholefeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (len(self.loader.images), feat_batch.shape[1], feat_batch.shape[2], feat_batch.shape[3])
                    features_size[i] = size_features
                    if memmap:
                        wholefeatures[i] = np.memmap(mmap_files[i], dtype=float, mode='w+', shape=size_features)
                    else:
                        wholefeatures[i] = np.zeros(size_features)
                np.save(features_size_file, features_size)

            start_idx = batch_idx*settings.BATCH_SIZE
            end_idx = min((batch_idx+1)*settings.BATCH_SIZE, len(self.loader.images))

            for i, feat_batch in enumerate(features_blobs):
                wholefeatures[i][start_idx:end_idx] = feat_batch

        return wholefeatures, size_features

    def quantile_threshold(self, features, savepath=''):
        qtpath = os.path.join(settings.OUTPUT_FOLDER, savepath)
        if savepath and os.path.exists(qtpath):
            return np.load(qtpath)
        print("calculating quantile threshold")
        quant = vecquantile.QuantileVector(depth=features.shape[1], seed=1)
        start_time = time.time()
        last_batch_time = start_time
        batch_size = 64
        for i in range(0, features.shape[0], batch_size):
            batch_time = time.time()
            rate = i / (batch_time - start_time + 1e-15)
            batch_rate = batch_size / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time
            print('Processing quantile index %d: %f %f' % (i, rate, batch_rate))
            batch = features[i:i + batch_size]
            batch = np.transpose(batch, axes=(0, 2, 3, 1)).reshape(-1, features.shape[1])
            quant.add(batch)
        #print('quantiling done')
        ret = quant.readout(1000)[:, int(1000 * (1-settings.QUANTILE)-1)]
        if savepath:
            np.save(qtpath, ret)
        return ret
        # return np.percentile(features,100*(1 - settings.QUANTILE),axis=axis)


    def tally(self, features, fsize, thresholds, use_crop_points=False):

        # print('hey')5 o Clock_Shadow, Arched Eyebrows, Bushy Eyebrows, No Beard, Bags Under_Eyes, Big Lips, Big Nose, Double Chin, Eyeglasses, Goatee, High Cheekbones, Rosy Cheeks, Mouth Slightly Open, Mustache, Narrow Eyes, Pointy Nose, Smiling, Wearing Lipstick

        num_units = features.shape[1]
        num_images = features.shape[0]
        # categories = self.loader.categories

        labels_map = dict((x, i) for i, x in enumerate(self.loader.dense_labels))

        def sample_tally(sample):

            for unit_idx in range(num_units):

                threshold = thresholds[unit_idx]
                img_idx = sample['i']
                # print(img_idx)
                label_set = sample['labels']
                # print(img_idx)
                # print(sample['fn'])
                fmap = features[img_idx, unit_idx, :, :]

                fmap = cv2.resize(fmap, (settings.INPUT_SIZE, settings.INPUT_SIZE))
                # print(threshold, fmap.min(), fmap.max())

                if fmap.max() > threshold:
                    # print('Hi', img_idx)

                    indexes = (fmap > threshold).astype(int)
                    thresh_fmap = fmap * indexes
                    # print(concept)
                    # cv2.imshow('i', thresh_fmap)
                    # cv2.waitKey(0)
                    for lab in label_set:

                        label_map = label_set[lab]
                        label_map = np.squeeze(label_map.numpy().transpose(1, 2, 0), axis=-1)
                        sub_cat = lab
                        # print(sub_cat)
                        # print(label_map.shape, label_map.max())
                        intersection = np.logical_and(label_map, thresh_fmap)
                        union = np.logical_or(label_map, thresh_fmap)
                        iou = (np.sum(intersection) / np.sum(union))
                        # print(intersection)
                        tally_units_int[unit_idx, labels_map[sub_cat]] += np.sum(intersection)
                        tally_units_uni[unit_idx, labels_map[sub_cat]] += np.sum(union)

                        all_maps[unit_idx][sub_cat].append(img_idx)
                        all_scores[unit_idx][sub_cat].append(iou)

                        # top10 = list(units_top10_score[unit_idx][labels_map[sub_cat]])
                        top20 = list(units_top20_score[unit_idx][labels_map[sub_cat]])

                        if iou > min(top20) and iou != 0:

                            units_top20_score[unit_idx][labels_map[sub_cat]][top20.index(min(top20))] = iou
                            units_top20_fmaps[unit_idx][labels_map[sub_cat]][top20.index(min(top20))] = img_idx


        # print(labels_map)
        if os.path.exists('{}/{}_tally20.npz'.format(settings.OUTPUT_FOLDER, settings.FEATURE_NAMES[0])):
            numpy_file = np.load(settings.OUTPUT_FOLDER + '/{}_tally20.npz'.format(settings.FEATURE_NAMES[0]))
            final_tally = numpy_file['tally']
            units_top20_fmaps = numpy_file['maps']
            units_top20_score = numpy_file['scores']

        else:

            tally_units_int = np.zeros((num_units, len(self.loader.dense_labels)), dtype=np.float64)
            tally_units_uni = np.zeros((num_units, len(self.loader.dense_labels)), dtype=np.float64)

            all_maps = dict((x, dict((y, []) for y in self.loader.dense_labels)) for x in range(num_units))
            all_scores = dict((x, dict((y, []) for y in self.loader.dense_labels)) for x in range(num_units))

            # units_top10_score = np.ones((num_units, len(self.loader.dense_labels), 10)) * -1
            # units_top10_fmaps = np.ones((num_units, len(self.loader.dense_labels), 10)) * -1

            units_top20_score = np.ones((num_units, len(self.loader.dense_labels), 20)) * -1
            units_top20_fmaps = np.ones((num_units, len(self.loader.dense_labels), 20)) * -1

            # tally_units = np.zeros(units,dtype=np.float64)
            # tally_units_cat = np.zeros((units,len(categories)), dtype=np.float64)
            # #print(tally_units_cat.shape)
            # tally_labels = np.zeros(labels,dtype=np.float64)

            if use_crop_points:
                sample_gen = self.tally_loader.get_batches(face_crop=True)
            else:
                sample_gen = self.tally_loader.get_batches()

            for idx in tqdm(range(0, num_images, settings.TALLY_BATCH_SIZE)):

                # print(idx)
                # start_time = time.time()
                sample_batch = next(sample_gen)
                # print(sample_batch)
                # curr_time = time.time()
                # fetch_time = curr_time - start_time
                # print('FETCH_TIME:', fetch_time)

                with concurrent.futures.ThreadPoolExecutor() as exec:

                    exec.map(sample_tally, sample_batch)


            # print(tally_units_iou[0])
            check = np.all((tally_units_uni == 0), axis=1)
            for i, value in enumerate(check):
                if value:
                    tally_units_uni[i] = 1

            # print(tally_units_occ[0])
            final_tally = tally_units_int / tally_units_uni
            # print(final_tally)

            np.savez(settings.OUTPUT_FOLDER + '/{}_tally20.npz'.format(settings.FEATURE_NAMES[0]),
                     tally=final_tally, maps=units_top20_fmaps, scores=units_top20_score) #, all_map=all_maps, all_sc=all_scores)

        # with open('{}/{}_all_maps.pickle'.format(settings.OUTPUT_FOLDER, settings.FEATURE_NAMES[0]), 'wb') as handle:
        #     pickle.dump(all_maps, handle, protocol=pickle.HIGHEST_PROTOCOL)

        sorting_ious = []
        sorted_features = np.zeros(fsize[0])
        sorted_final_tally = np.zeros(final_tally.shape)
        sorted_units_top20_score = np.zeros(units_top20_score.shape)
        sorted_units_top20_fmaps = np.zeros(units_top20_fmaps.shape)
        sorted_thresholds = np.zeros(thresholds.shape)
        print('Sorting features IOU-wise')

        for ui in range(num_units):
            us = final_tally[ui]
            sci = np.nanargmax(us)
            cc = self.loader.dense_labels[sci]
            iou = us[sci]
            sorting_ious.append(iou)

        sorted_indexes = np.argsort(sorting_ious)[::-1]

        for ii, k in enumerate(sorted_indexes):
            sorted_features[:, ii, :, :] = features[:, k, :, :]
            sorted_thresholds[ii] = thresholds[k]
            sorted_final_tally[ii] = final_tally[k]
            sorted_units_top20_fmaps[ii] = units_top20_fmaps[k]
            sorted_units_top20_score[ii] = units_top20_score[k]


        if use_crop_points:
            face_mod = MTCNN(image_size=256)

        # count = 0
        # total = 0

        print('Drawing Top Features per Unit')

        with PdfPages(settings.OUTPUT_FOLDER + '/{}.pdf'.format(settings.FEATURE_NAMES[0])) as pdf:
            num_top_images = settings.TOPN
            for ux in range(0, num_units, 4):
                fig = plt.figure()
                fig, ax = plt.subplots(4, 1)
                if ux == 0:
                    fig.suptitle('Face NetDissect Results')
                for k in range(0, 4):
                    threshold = sorted_thresholds[ux + k]
                    unit_scores = sorted_final_tally[ux + k]
                    # unit_scores[unit_scores == np.nan] = -1
                    # if unit_scores.max() == 0:
                    #     continue
                    # print(type(unit_scores), unit_scores)
                    subcat_idx = np.nanargmax(unit_scores)
                    curr_class = self.loader.dense_labels[subcat_idx]
                    # print(subcat_idx)
                    overall_iou = unit_scores[subcat_idx]
                    # if overall_iou > 0.04:
                    #     count += 1
                    #     total += overall_iou

                    # scores = units_top4_score[ux][subcat_idx]
                    units_top_idx = np.argsort(sorted_units_top20_score[ux + k][subcat_idx])[-num_top_images:]
                    fmap_idxs = [sorted_units_top20_fmaps[ux + k][subcat_idx][x] for x in units_top_idx]
                    # print(fmap_idxs)
                    fmap_idxs = [int(x) for x in fmap_idxs if int(x) != -1]
                    tile = np.zeros((settings.INPUT_SIZE, settings.INPUT_SIZE * num_top_images, 3), dtype=np.uint8) # here
                    if len(fmap_idxs) > 0:
                        # print('hi')
                        fmaps = [cv2.resize(sorted_features[x][ux + k], (settings.INPUT_SIZE, settings.INPUT_SIZE)) for x in fmap_idxs]

                        images = [cv2.resize(cv2.imread('visual_dictionary/' + self.loader.images[x]['image']),
                                  (settings.INPUT_SIZE, settings.INPUT_SIZE)) for x in fmap_idxs] # here

                        pil_images = [Image.open('visual_dictionary/' + self.loader.images[x]['image']).convert('RGB')
                                      .resize((settings.INPUT_SIZE, settings.INPUT_SIZE)) for x in fmap_idxs] # here

                        crop_images = [None] * len(images)
                        if use_crop_points:
                            # print(ux+k, [x.mode for x in pil_images])

                            points = [face_mod.detect(x.copy(), landmarks=False) for x in pil_images]
                            points = [x[0] for x in points]
                            new_points = [None] * len(points)
                            NoneType = type(None)
                            for ix, pt in enumerate(points):
                                if isinstance(pt, NoneType):
                                    # print('hi')
                                    new_pt = np.array([0, 0, settings.INPUT_SIZE, settings.INPUT_SIZE])# here
                                    new_points[ix] = new_pt
                                    curr_img = cv2.resize(images[ix], (settings.INPUT_SIZE, settings.INPUT_SIZE))
                                    crop_images[ix] = curr_img
                                else:
                                    # print('hi')
                                    new_pt = pt[0]
                                    new_pt = [int(x) for x in new_pt]
                                    new_pt[0] = max(0, new_pt[0])
                                    new_pt[1] = max(0, new_pt[1])
                                    new_pt[2] = min(settings.INPUT_SIZE, new_pt[2]) # here
                                    new_pt[3] = min(settings.INPUT_SIZE, new_pt[3]) # here
                                    new_points[ix] = new_pt
                                    # print(images[ix].shape, new_pt)
                                    crop_images[ix] = cv2.resize(images[ix][new_pt[1]:new_pt[3], new_pt[0]:new_pt[2]],
                                                                (settings.INPUT_SIZE, settings.INPUT_SIZE))

                        for i in range(len(fmaps)):
                            if use_crop_points:
                                img = crop_images[i]
                            else:
                                img = images[i]
                                # img = cv2.resize(img, (settings.INPUT_SIZE, settings.INPUT_SIZE))
                            img_copy = img.copy()
                            output = img.copy()
                            # print(fmaps[i].min())
                            indexes = (fmaps[i] > threshold).astype(int)
                            thresh_map = fmaps[i] * indexes
                            # print(img.shape, thresh_map.shape)
                            img_copy[thresh_map <= 0] = 0
                            # cv2.imshow('i', img)
                            # cv2.imshow('k', thresh_map)
                            # cv2.waitKey(0)
                            cv2.addWeighted(img, 0.4, img_copy, 0.6, 0, output)
                            # cv2.imshow('o', output)
                            # cv2.waitKey(0)
                            if use_crop_points:
                                full_img = images[i].astype(np.uint8)
                                overlay = np.zeros((full_img.shape)).astype(np.uint8)
                                # print(full_img.shape, overlay.shape, full_output.shape)
                                full_output = cv2.addWeighted(full_img, 0.4, overlay, 0.6, 0)
                                x1, y1, x2, y2 = new_points[i]
                                output = cv2.resize(output, (x2-x1, y2-y1))
                                # print(x2-x1, y2-y1, output.shape)
                                full_output[y1:y2, x1:x2] = output
                                output = full_output
                            tile[:, settings.INPUT_SIZE * 2 * i: settings.INPUT_SIZE * 2 * (i+1), :] = output # here
                            # cv2.imshow('i', fmaps[i])
                            # cv2.waitKey(0)
                        ax[k].imshow(tile[:,:,::-1])
                        ax[k].set_title('UNIT - {},  CLASS - {}, IOU - {:.4f}'.format(sorted_indexes[ux + k], curr_class, overall_iou), fontsize=7)
                        ax[k].axes.xaxis.set_visible(False)
                        ax[k].axes.yaxis.set_visible(False)
                    else:
                        # print('bye')
                        ax[k].imshow(tile[:,:,::-1])
                        ax[k].set_title('UNIT - {},  CLASS - NONE, IOU - N/A'.format(sorted_indexes[ux + k]), fontsize=7)
                        ax[k].axes.xaxis.set_visible(False)
                        ax[k].axes.yaxis.set_visible(False)
                pdf.savefig()
                plt.close()
