import logging
import os
import random

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from datasets import Dataset 
import numpy as np
import pandas as pd
import spacy
from torch.utils.tensorboard import SummaryWriter
import torch
from transformers import GPT2Tokenizer

from src.full_model.my_evaluate import plot_detections_and_sentences_to_tensorboard, get_generated_reports, update_gen_sentences_with_corresponding_regions
from src.full_model.report_generation_model import ReportGenerationModel

BERTSCORE_SIMILARITY_THRESHOLD = 0.9
IMAGE_INPUT_SIZE = 512
MAX_NUM_TOKENS_GENERATE = 300
NUM_BEAMS = 4
NUM_IMAGES_TO_PLOT = 8
SEED = 42
mean = 0.471 
std = 0.302

# Set GPU Check 
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Set the CUDA device to use (e.g., GPU with index 0)
    device = torch.device("cuda")   # Now you can use CUDA operations
else:
    device = torch.device("cpu")

logging.basicConfig(level=logging.INFO, format="[%(levelname)s]: %(message)s")
log = logging.getLogger(__name__)

# set the seed for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def get_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_data_details(row):
    image = cv2.imread(row, cv2.IMREAD_UNCHANGED)  
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    

    val_test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )

    transform = val_test_transforms(image=image)
    image_transformed = transform["image"]  # shape (1, 512, 512)
    image_transformed_batch = image_transformed.unsqueeze(0)

    usecols = [
        "mimic_image_file_path",
        "reference_report"
    ]
    dataset_as_df = {}
    dataset_as_df["test"] = pd.read_csv(os.path.join("../Radiology-Report-Generation---MGA/dataset-with-reference-reports/test-demo.csv"), usecols=usecols)
    # Find the index of "/images" - as in csv only relative paths are stored
    index = row.find("/images")
    # Extract the substring from "/images" onwards
    row = row[index:]
    dataset_as_df["test"] = dataset_as_df["test"] [dataset_as_df["test"]['mimic_image_file_path'] == row]
    total_num_samples_val = len(dataset_as_df["test"])
    print("Num of Loaded Images: ", total_num_samples_val)
    reference_report = dataset_as_df["test"]["reference_report"]

    return image_transformed_batch, reference_report


def infer(row):

    image_tensor,reference_report = get_data_details(row)
    tokenizer = get_tokenizer()

    checkpoint = torch.load(
        "../Radiology-Report-Generation---MGA/runs/full_model/run_122/checkpoints/checkpoint_val_loss_22.038_overall_steps_14010.pt",
        map_location=torch.device(device=device),
    )
    
    tensorboard_folder_path = "../Radiology-Report-Generation---MGA/Inferences - Sample Reports Generated/DEMO_Tensorboard"

    writer = SummaryWriter(log_dir=tensorboard_folder_path)
    
    # if there is a key error when loading checkpoint, try uncommenting down below
    # since depending on the torch version, the state dicts may be different
    model = ReportGenerationModel(pretrain_without_lm_model=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device, non_blocking=True)
    model.eval()
 
    del checkpoint

    #Report Gen Model output

    sentence_tokenizer = spacy.load("en_core_web_trf")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model.generate(
            image_tensor.to(device, non_blocking=True),
            max_length=MAX_NUM_TOKENS_GENERATE,
            num_beams=NUM_BEAMS,
            early_stopping=True,
        )

    beam_search_output, selected_regions, detections, class_detected = output
    
    selected_regions = selected_regions.detach().cpu().numpy()

    generated_sents_for_selected_regions = tokenizer.batch_decode(
        beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )  
    generated_sents_for_selected_regions, generated_reports, _ = get_generated_reports(
                                                                    generated_sents_for_selected_regions,
                                                                    selected_regions,
                                                                    sentence_tokenizer,
                                                                    BERTSCORE_SIMILARITY_THRESHOLD
                                                                )   
    gen_sentences_with_corresponding_regions = []
    update_gen_sentences_with_corresponding_regions(gen_sentences_with_corresponding_regions, generated_sents_for_selected_regions, selected_regions)

    gen_and_ref_reports = {
        "generated_reports": [],
        "reference_reports": [],
    }
    gen_and_ref_reports["generated_reports"].extend(generated_reports)
    gen_and_ref_reports["reference_reports"].extend(reference_report)

    #Object Detector Output
    print("NUMBER OF IMAGES PLOTTED: ", NUM_IMAGES_TO_PLOT)
    plot_detections_and_sentences_to_tensorboard(
        writer=writer,
        overall_steps_taken=10000,
        image_tensor=image_tensor,
        detections=detections,
        class_detected=class_detected
    )
    return (gen_sentences_with_corresponding_regions, gen_and_ref_reports)

if __name__ == "__main__":
    main()