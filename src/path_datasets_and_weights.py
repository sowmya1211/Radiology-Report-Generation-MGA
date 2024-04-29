"""
URL / Source for Dataset:

1. MIMIC-CXR 
   MIMIC-CXR-JPG (CXR Images in jpg format)  - https://physionet.org/content/mimic-cxr-jpg/2.1.0/
   Reference reports - https://physionet.org/content/mimic-cxr/2.0.0/

   MIMIC-CXR and MIMIC-CXR-JPG dataset paths should both have a (sub-)directory called "files" in their directories.
   Note that we only need the report txt files from MIMIC-CXR, which are in the file mimic-cxr-report.zip

2. Chest ImaGenome - https://physionet.org/content/chest-imagenome/1.0.0/
    Chest ImaGenome dataset path should have a (sub-)directory called "silver_dataset" in its directory.
"""

path_chest_imagenome = "{path_to_chest_imagenome_dataset_directory}" #For storing chest imagenome dataset - scene graph information; must have subdirectory "silver_dataset"
path_mimic_cxr = "{path_to_mimic_cxr_directory}" #For reference reports
path_mimic_cxr_jpg = "{path_to_mimic_cxr_jpg_directory}" #For CXR images in jpg format
path_full_dataset = "../dataset-with-reference-reports" #Holds the dataset split csv files created after processing the dataset
path_chexbert_weights = "CheXbert/src/models/chexbert.pth"
path_runs_object_detector = "../runs/object_detector" #To store checkpoints of first stage (object detector)  
path_runs_full_model = "../runs/full_model" #To store checkpoints of last 2 stages (full model with and without language model)
path_test_set_evaluation_scores_txt_files = "../runs/scores" #To store scores and generated sentences and reports during testing