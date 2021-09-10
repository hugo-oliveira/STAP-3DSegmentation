STAP-3DSegmentation
================

### Description
A pediatric brain MRI segmentation pipeline composed of preprocessing, semantic segmentation and post-processing steps. The preprocessing step consists of Brain Extraction followed by Bias Field Correction. The pipeline follows with the segmentation module composed of two ensembles of networks: generalists and specialists. The generalist networks are responsible for locating and roughly segmenting the brain areas, yielding regions of interest for each target organ. Specialist networks can then improve the segmentation performance for underrepresented organs by learning only from the regions of interest from the generalist networks. At last, post-processing consists in merging the specialist and generalist networks predictions and performing late fusion across the distinct architectures to generate a final prediction.


Table of contents
=================
<!--ts-->
   * [Description](#Description)
   * [Table of contents](#tabela-de-conteudo)
   * [Usage](#Usage)
      * [Prerequisites](#prerequisites)
      * [Local Files](#local-files)
      * [Experiment Configuration](#experiment-configuration)
      * [Labels and weights Configuration](#labels-and-weights-configuration)
   * [Tests](#testes)
<!--te-->

# Usage

## Prerequisites

Before to begin you will need to install in your machine the following tools:

[Git](https://git-scm.com), [Python](https://www.python.org/) and any volumetric visualisation tool like [3D-Slice](https://www.slicer.org/). 

After install python in your machine you will need to install all packages used by this project:
```
# clone this repository
git clone https://github.com/hugo-oliveira/STAP-3DSegmentation.git

# You can use pip
$ pip install -r requeriments.txt
```

## Local Files
You can use your own dataset or our dataset availaible in [link](https://drive.google.com/drive/folders/1scogXY_iKhGz6CzY4196uFCKJK8iXvVc?usp=sharing)  and place the .nii.gz files in the directory `datasets/hc_pediatric/images/` Ground truths must also be downloaded from [here](https://drive.google.com/drive/folders/1qV_WfjYMQ5rOOtWba3BrgvzlmgZSU-XX?usp=sharing) and placed on the directory `datasets/hc_pediatric/ground_truths/all/`. 
If you use your own dataset you will need to organize the files this way:

```bash
./YOUR_DATASET_NAME/
├── images/
└── ground_truths/
```

## Preprocessing
To perform a preprocessing step in the dataset run the following line:

```
$ preprocessing.sh YOUR_DATASET_NAME
```
This will create a new directory pre_YOUR_DATASET_NAME, use this directory to configurate the experiment (follow step).

## Labels and weights Configuration

Before perform a experiment you'll need to create two text file content the configuration of the labels and the weights of the network.  
   1. Create a file named `all_valid_labels.txt` and put it inside `YOUR_DATASET_NAME` and organize the content as follows (Examples can be found inside our dataset `datasets/hc_pediatric/all_valid_labels.txt`):
   ```
   class_name_1: LABEL_NUM_IN_GT->CLASS_NUMBER
   class_name_2: LABEL_NUM_IN_GT->CLASS_NUMBER2
   ...
   class_name_n: LABEL_NUM_IN_GT->CLASS_NUMBERN
   ```

   2. Create a file named `all_weights.txt` and put it inside `YOUR_DATASET_NAME` and organize the content as follows (Examples can be found inside our dataset `datasets/hc_pediatric/all_weights.txt`):
   ```
   INT_WEIGHT_CLASS_NUMBER
   INT_WEIGHT_CLASS_NUMBER2
   ...
   INT_WEIGHT_CLASS_NUMBERN
   ```

## Experiment
1. A randomly selected split for the data is provided in the files `datasets/hc_pediatric/all_trn_f*.txt` and `datasets/hc_pediatric/all_tst_f*.txt`, however, with the addition of new data, one can randomly generate folds and overwrite these files using the script `split_batches.py`. This python program splits the train and test samples according to the files in the ground truth directory.
```
$ python split_batches ./YOUR_DATASET_DIR
```

2. Before perform an experiment you will need to create a configurate file. Some examples you can find in "./config/". After that you can run a experiment using the following line:

```
$ nohup python train.py ./config/YOUR_CONFIG_FILE > LOG_EXPERIMENT_NAME.log
```


## Visuals
- Preprocessing step:

![Alt Text](./imagens/preprocessing.png)

- Segmentation and fusion step:



- Some results:
