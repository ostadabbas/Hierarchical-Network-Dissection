import settings
from feature_operation import hook_feature,FeatureOperator
from visualize.report import generate_html_summary
from util.clean import clean
import numpy as np

if __name__=='__main__':

    if settings.TASK == 'age':
        from loader.model_loader_age import loadmodel
    elif settings.TASK == 'gender':
        from loader.model_loader_age import loadmodel
    elif settings.TASK == 'beauty':
        from loader.model_loader_age import loadmodel
    elif settings.TASK == 'facenet':
        from loader.model_loader_age import loadmodel
    elif settings.TASK == 'fairface':
        from loader.model_loader_age import loadmodel
    if settings.TASK == 'smile':
        from loader.model_loader_age import loadmodel
    else:
        print('Invalid Model Choice, Retry with valid task')
        print('Currently supported values:')
        for task in list(settings.SIZES.keys()):
            print(task)


    fo = FeatureOperator()
    model = loadmodel(hook_feature)

    ############ STEP 1: feature extraction ###########self.loader.num_batches####
    features, size = fo.feature_extraction(model=model, face_crop=settings.CROP)

    for layer_id,layer in enumerate(settings.FEATURE_NAMES):
    ############ STEP 2: calculating threshold ############
        thresholds = fo.quantile_threshold(features[layer_id],savepath="{}_quantile.npy".format(layer))
        # print(thresholds)

    # ############ STEP 3: calculating IoU scores and generate report ###########

        tally_result = fo.tally(features[layer_id], size, thresholds, use_crop_points=settings.CROP)

