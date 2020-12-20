# Face Dissect

This is the official pytorch implementation of Face Dissect which performs network dissection on several face models as described in the [paper](). Also, this repo contains the link to the first ever Face Dictionary that contains several face concepts annotated under the same dataset.

# Prerequisites:

- Torch=1.5.0
- PIL
- opencv
- numpy
- matplotlib
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)
- cuda/cudnn (recommended)
- tqdm

## Preparation:

- **Data**
    - Download the data here(link to dictionary) and put in viusal_dictionary folder.
    - Download the models from here (link to models) and put them in model folder.
- **Code**
    - Change flags in settings.py to determine which model to dissect.
    - Change the name of the layer in model loader scripts to choose which layer to dissect.


## How to run:

### Stage 1 (Dissection)
The first stage of the process requires us to extract the activation maps of all the images (N) in the Face Dictionary with labelled local concepts for the layer specified in the chosen model loader script accordingly. For all the units (U) in the given layer, we generate N maps from the `main.py` script which first stores the activations in a numpy memory map. The size of the memory map is smaller for the deeper layers due to the lower resolutions of the activation map (HxW). The first function 'feature_extraction' generates two files and stores them in the auto generated 'result' folder that are:
- layer.mmap
- layer_feature_size.npy

Secondly after generating the features, we now estimate the threshold values per unit by using their spatial features and computing a value such that the probability of any spatial location having a value greater than the threshold is equal to 0.005 (99.5 quantile). The second function called 'quantile_threshold' computes that and these values are stored in a file called:
- layer_quantile.npy

Finally, we use these thresholds to segment the activation maps for each unit respectively and evaluate them against the binary labels we have assembled in the dictionary per concept to compute their intersections and unions with them. Only those images that have a labelled instance of a given concept are used for this evaluation and once we iterate through the entire dictionary, we obtain a list of intersections and unions for each unit that has a length eqaul to the number of concepts. Then we divide the intersections by the unions and generate a final dataset wide IoU for each unit-concept pair. The concept with the highest IoU is recorded and the unit is said to be interpretable if the top concept has an IoU > 0.04. The third function 'tally' performs this computation and generates a pdf file (as shown below) that displays the top four images with the highest IoU returned for that concept per unit. This function generates two files that record the IoU values and display the dissection report respectively which are:
- layer_tally.npz
- layer_conv.pdf

![plot](imgs/report_photo.png)

### Stage 2 (Probabilistic Hierarchy)

Even though we can obtain the dominant concepts per unit based on IoU, more often than not there is more than one concept that manages to obtain a high IoU and it is very likely these concepts lie in a similar region of the face. In that case, it is better to establish a hierarchy of concepts that lie in the same region of the face as that of the top concept returned by Stage 1 pipeline. In order to do that we run the `cluster_top.py` script as it identifies the facial region and then generates probabilities for every concept within that region of the face. This script generates the probabilities in the form of a text file (as shown below) named:
- layer_top_concept_probs.txt

![plot](imgs/cluster_probs.png)

### Stage 3 (Non Localizable Bias)

In order to determine if any of the units in the dissected layer has a bias towards any of the 4 non local concepts we have provided in the dictionary. Using the tally file generated in Stage 1, it uses our formulation as described in the [paper]() to formulate a probability for each sub group of the non local concept exactly in the same manner as stage 2. Run the `nl_bias.py` script to generate a probability text file for each task that has been dissected (alter the 'tasks' list defined in the script to include only those task models that have been dissected using Stage1) and the script will genrate a set of features and a probability text file for each non local concept named as follows:
- layer_model_bias(skin_tone).npz
- layer_model_bias(ethnic).npz
- layer_model_bias(gender).npz
- layer_model_bias(age).npz
- layer_NL_concept_probs.txt
