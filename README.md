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

### Preparation:

- **Data**
    - Download the data here(link to dictionary) and put in viusal_dictionary folder.
    - Download the models from here (link to models) and put them in model folder.
- **Code**
    - Change flags in settings.py to determine which model to dissect.
    - Change the name of the layer in model loader scripts to choose which layer to dissect.


### How to run:

1. **Dissection:** Run the main.py script to execute IoU based dissection and generate a report which will be saved in an auto generated result folder.
2. **Probability Scores:** Run the cluster_top.py script to generate a probability distribution of all the concepts that belong to the cluster of the top IoU concept per unit returned by dissection in a text file.
3. **Non Localizable Bias Analysis:** Run the nl_bias.py script to generate features for the non local set in dictionary for all the models and display their relative probabilities for each non local concept in a text file. (Warning: Huge files will be generated for upper layers with higher resolution).
