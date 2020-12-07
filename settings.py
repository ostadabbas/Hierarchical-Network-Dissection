######### global settings  #########
GPU = True                                  # running on GPU is highly suggested
MODEL = 'resnet50'                          # model arch: resnet18, alexnet, resnet50, densenet161, etc.
DATASET = 'IMDB-WIKI'                       # model trained on: vggface, SCUT-5500, etc.
TASK = 'age'                                # choose from 'age', 'gender', 'beauty', 'facenet', 'fairface', 'smile'
LAYER_NAME = 'layer4'                       # the layer that is being dissected , ex. layer4, layer3, layer2, layer1
QUANTILE = 0.05                             # the threshold used for activation 0.005
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
SCORE_THRESHOLD = 0.04                      # the threshold used for IoU score
TOPN = 4                                    # to show top N image with highest IoU for each unit
CATAGORIES = ["AU", "attr", "parts"]        # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
OUTPUT_FOLDER = "result/Localizable"        # result will be stored in this folder
NL_FOLDER = "result/Non_Localizable"
FEATURE_NAMES = ["{}_{}".format(TASK, )]
SIZES = {'age':128, 'gender':128, 'beauty':224, 'facenet':224, 'fairface':112, 'smile':200}
INPUT_SIZE = SIZES[TASK]
# WORKERS = 1
BATCH_SIZE = 8
TALLY_BATCH_SIZE = 32
# TALLY_AHEAD = 4
# INDEX_FILE = 'final_dict.csv'
CROP = True
