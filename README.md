# Radiology Report Generation - MGA
 Automatic Report Generation from Medical Images using Memory-Guided Attention





## Method

Radiology Report Generation using **M**emory-**G**uided **A**ttention (MGA): 
Transformer-based encoder decoder system - <br>
Encoder detects and extracts relevant regions of interest (RoIs) and region-specific visual features <br>
Decoder is based on Generative Pre-trained Transformer (GPT2) with pseudo-attention layers and is integrated with a Relational Memory (RM).

## Setup

1. Create conda environment with "**conda env create -f environment.yml**"
2. Install Java 1.8.0 (required for pycocoevalcap library, see https://pypi.org/project/pycocoevalcap/). On Ubuntu, you can install Java 1.8.0 with "**sudo apt install openjdk-8-jdk**".

## Datasets and Checkpoints
1. URL / Source for Dataset:

  - Refer to the following links to download the required datasets; they are publicly available from PhysioNet but require credentialized access 

  - MIMIC-CXR 
    CXR Images in jpg format - https://physionet.org/content/mimic-cxr-jpg/2.1.0/ <br>
    Reference reports - https://physionet.org/content/mimic-cxr/2.0.0/
    
  - Chest ImaGenome - https://physionet.org/content/chest-imagenome/1.0.0/

2. Download CheXbert model checkpoint from [link](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) <br>
   Store the checkpoint in this directory [CheXbert_checkpoint](src/CheXbert/src/models)

3. Download the full model checkpoint from this [google drive link](https://drive.google.com/file/d/1x6yjKv7CbbCI2xa2tAcjaIYMVBIjV2mH/view?usp=sharing) <br>
   Store the checkpoint in this directory [full_model_checkpoint](runs/full_model/run_122/checkpoints)

4. Download the object detector model checkpoint from this [google drive link](https://drive.google.com/file/d/1E5ky_khhzhVXXUslZ4OiSU_Ab1MjoFlq/view?usp=sharing) <br>
   Store the checkpoint in this directory [obj_detector_checkpoint](runs/object_detector/run_1/weights)

4. In [path_datasets_and_weights.py](src/path_datasets_and_weights.py), follow the instructions to download specific folders from the datasets links <br>
   Specify the paths to the various datasets (Chest ImaGenome, MIMIC-CXR, MIMIC-CXR-JPG), CheXbert weights, and other important folders

## Create dataset split
After the setup, run "**python [create_dataset.py](src/dataset/create_dataset.py)**" to create train, val and test csv files, in which each row contains specific information about a single image. See doc string of create_dataset.py for more details.

## Training and Testing

Please read [README_TRAIN_TEST.md](README_TRAIN_TEST.md) for specific information on training and testing the model.

## Inference

To generate reports for a list of images, run "**python [generate_reports_for_images.py](src/full_model/generate_reports_for_images.py)**".  <br>
Specify the model checkpoint, the list of image paths and the path to the txt file with the generated reports in the main function.

## Running Web App

To run the web app, run "**python [app.py](src/web-app/app.py)**".  <br>
Specify the model checkpoint, in [my_inference.py](src/full_model/my_inference.py)".
