from ast import literal_eval
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from torch.utils.data import DataLoader 
from transformers import GPT2Tokenizer
from tqdm import tqdm

from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.custom_collator import CustomCollator
from src.full_model.custom_dataset import CustomDataset
from src.full_model.my_evaluate import get_generated_reports, get_ref_sentences_for_selected_regions, get_sents_for_normal_abnormal_selected_regions, plot_detections_and_sentences_to_tensorboard, update_gen_and_ref_sentences_for_regions, update_gen_sentences_with_corresponding_regions, update_num_generated_sentences_per_image, write_sentences_and_reports_to_file
from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.run_configurations import (
    RUN,
    RUN_COMMENT,
    SEED,
    PRETRAIN_WITHOUT_LM_MODEL,
    IMAGE_INPUT_SIZE,
    PERCENTAGE_OF_TRAIN_SET_TO_USE,
    PERCENTAGE_OF_VAL_SET_TO_USE,
    BATCH_SIZE,
    EFFECTIVE_BATCH_SIZE,
    NUM_WORKERS,
    EPOCHS,
    LR,
    EVALUATE_EVERY_K_BATCHES,
    PATIENCE_LR_SCHEDULER,
    THRESHOLD_LR_SCHEDULER,
    FACTOR_LR_SCHEDULER,
    COOLDOWN_LR_SCHEDULER,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
    NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
    NUM_IMAGES_TO_PLOT,
    BERTSCORE_SIMILARITY_THRESHOLD,
    WEIGHT_OBJECT_DETECTOR_LOSS,
    WEIGHT_BINARY_CLASSIFIER_REGION_SELECTION_LOSS,
    WEIGHT_BINARY_CLASSIFIER_REGION_ABNORMAL_LOSS,
    WEIGHT_LANGUAGE_MODEL_LOSS,
)
from src.path_datasets_and_weights import path_full_dataset, path_runs_full_model

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

def get_data_loaders(tokenizer, val_dataset):
    def seed_worker(worker_id):
        """To preserve reproducibility for the randomly shuffled train loader."""
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    #custom_collate_train = CustomCollator(tokenizer=tokenizer, is_val_or_test=False, pretrain_without_lm_model=PRETRAIN_WITHOUT_LM_MODEL)
    custom_collate_val = CustomCollator(tokenizer=tokenizer, is_val_or_test=True, pretrain_without_lm_model=PRETRAIN_WITHOUT_LM_MODEL)

    g = torch.Generator()
    g.manual_seed(SEED)

    val_loader = DataLoader(
        val_dataset,
        collate_fn=custom_collate_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # could also be set to NUM_WORKERS, but I had some problems with the val loader stopping sometimes when num_workers != 0
        pin_memory=True,
    )

    return val_loader


def get_transforms(dataset: str):
    # see compute_mean_std_dataset.py in src/dataset
    mean = 0.471
    std = 0.302

    # use albumentations for Compose and transforms
    # augmentations are applied with prob=0.5
    # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
    train_transforms = A.Compose(
        [
            # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
            # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
            # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
            # INTER_AREA works best for shrinking images
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            # randomly (by default prob=0.5) translate and rotate image
            # mode and cval specify that black pixels are used to fill in newly created pixels
            # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0, translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    # don't apply data augmentations to val set (and test set)
    val_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE, min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

    if dataset == "train":
        return train_transforms
    else:
        return val_transforms


def get_tokenized_datasets(tokenizer, raw_val_dataset):
    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    #tokenized_train_dataset = raw_train_dataset.map(tokenize_function)
    tokenized_val_dataset = raw_val_dataset.map(tokenize_function)

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path (str)
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])
    #
    #   val dataset will have additional column:
    #   - reference_report (str)

    return tokenized_val_dataset


def get_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_datasets(row):
    usecols = [
        "mimic_image_file_path",
        "bbox_coordinates",
        "bbox_labels",
        "bbox_phrases",
        "bbox_phrase_exists",
        "bbox_is_abnormal",
    ]

    # all of the columns below are stored as strings in the csv_file
    # however, as they are actually lists, we apply the literal_eval func to convert them to lists
    converters = {
        "bbox_coordinates": literal_eval,
        "bbox_labels": literal_eval,
        "bbox_phrases": literal_eval,
        "bbox_phrase_exists": literal_eval,
        "bbox_is_abnormal": literal_eval,
    }

    datasets_as_dfs = {}

    # val dataset has additional "reference_report" column
    usecols.append("reference_report")
    datasets_as_dfs["valid"] = pd.read_csv(os.path.join("../rgrg+mdt/dataset-with-reference-reports/test.csv"), usecols=usecols, converters=converters)
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"] [datasets_as_dfs["valid"]['mimic_image_file_path'] == row]
    total_num_samples_val = len(datasets_as_dfs["valid"])
    print("loaded", total_num_samples_val)

    # compute new number of samples for both train and val
    #new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)

    log.info(f"Val: {new_num_samples_val} images")

    # limit the datasets to those new numbers
    datasets_as_dfs["valid"] = datasets_as_dfs["valid"][:new_num_samples_val]

    raw_val_dataset = Dataset.from_pandas(datasets_as_dfs["valid"])

    return raw_val_dataset #raw_train_dataset, raw_val_dataset


def infer(row):
    # the datasets still contain the untokenized phrases
    raw_val_dataset = get_datasets(row)

    tokenizer = get_tokenizer()

    # tokenize the raw datasets
    tokenized_val_dataset = get_tokenized_datasets(tokenizer, raw_val_dataset)

    val_transforms = get_transforms("val")

    val_dataset_complete = CustomDataset("val", tokenized_val_dataset, val_transforms, log)

    val_dl = get_data_loaders(tokenizer, val_dataset_complete)

    checkpoint = torch.load(
        "/home/miruna/ReportGeneration_SSS_24/rgrg+mdt/runs/full_model/run_122/checkpoints/checkpoint_val_loss_22.038_overall_steps_14010.pt",
        map_location=torch.device(device=device),
    )
    
    tensorboard_folder_path = "/home/miruna/ReportGeneration_SSS_24/rgrg+mdt/Inferences - Sample Reports Generated/DEMO_Tensorboard"

    writer = SummaryWriter(log_dir=tensorboard_folder_path)
    
    # if there is a key error when loading checkpoint, try uncommenting down below
    # since depending on the torch version, the state dicts may be different
    model = ReportGenerationModel(pretrain_without_lm_model=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device, non_blocking=True)
    model.eval()

    del checkpoint
    
    model.eval()
    
    gen_and_ref_sentences = {
        "generated_sentences": [],
        "generated_sentences_normal_selected_regions": [],
        "generated_sentences_abnormal_selected_regions": [],
        "reference_sentences": [],
        "reference_sentences_normal_selected_regions": [],
        "reference_sentences_abnormal_selected_regions": [],
        "num_generated_sentences_per_image": []
    }
    
    for region_index, _ in enumerate(ANATOMICAL_REGIONS):
        gen_and_ref_sentences[region_index] = {
            "generated_sentences": [],
            "reference_sentences": []
        }
        
    gen_and_ref_reports = {
        "generated_reports": [],
        "removed_similar_generated_sentences": [],
        "reference_reports": [],
    }
     
    gen_sentences_with_corresponding_regions = []
    num_batches_to_process_for_image_plotting = NUM_IMAGES_TO_PLOT // BATCH_SIZE
    print("NUMBER OF IMAGES PLOTTED: ", num_batches_to_process_for_image_plotting, NUM_IMAGES_TO_PLOT, NUM_IMAGES_TO_PLOT)
    
    sentence_tokenizer = spacy.load("en_core_web_trf")
    
    oom = False
    
    with torch.no_grad():
        for num_batch, batch in tqdm(enumerate(val_dl), total=NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION):
            # since generating sentences takes some time, we limit the number of batches used to compute bleu/rouge-l/meteor
            print(num_batch)
            if num_batch >= NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION:
                break

            images = batch["images"]  # shape [batch_size x 1 x 512 x 512]
            image_targets = batch["image_targets"]
            region_is_abnormal = batch["region_is_abnormal"].numpy()  # boolean array of shape [batch_size x 29]

            # List[List[str]] that holds the reference phrases. The inner list holds all reference phrases of a single image
            reference_sentences = batch["reference_sentences"]

            # List[str] that holds the reference report for the images in the batch
            reference_reports = batch["reference_reports"]

            try:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model.generate(
                        images.to(device, non_blocking=True),
                        max_length=MAX_NUM_TOKENS_GENERATE,
                        num_beams=NUM_BEAMS,
                        early_stopping=True,
                    )
            except RuntimeError as e:  # out of memory error
                if "out of memory" in str(e):
                    oom = True
                    
                    print(f"Generation:\nOOM at batch number {num_batch}.\nError Message: {str(e)}\n\n")

                else:
                    raise e

            if oom:
                # free up memory
                torch.cuda.empty_cache()
                oom = False
                continue

            # output == -1 if the region features that would have been passed into the language model were empty (see forward method for more details)
            if output == -1:
                print(f"Generation:\nEmpty region features before language model at batch number {num_batch}.\n\n")
                continue
            else:
                # selected_regions is of shape [batch_size x 29] and is True for regions that should get a sentence
                beam_search_output, selected_regions, detections, class_detected = output
                selected_regions = selected_regions.detach().cpu().numpy()

            # generated_sents_for_selected_regions is a List[str] of length "num_regions_selected_in_batch"
            generated_sents_for_selected_regions = tokenizer.batch_decode(
                beam_search_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )

            # filter reference_sentences to those that correspond to the generated_sentences for the selected regions.
            # reference_sents_for_selected_regions will therefore be a List[str] of length "num_regions_selected_in_batch"
            # (i.e. same length as generated_sents_for_selected_regions)
            reference_sents_for_selected_regions = get_ref_sentences_for_selected_regions(
                reference_sentences, selected_regions
            )

            (
                gen_sents_for_normal_selected_regions,
                gen_sents_for_abnormal_selected_regions,
                ref_sents_for_normal_selected_regions,
                ref_sents_for_abnormal_selected_regions,
            ) = get_sents_for_normal_abnormal_selected_regions(region_is_abnormal, selected_regions, generated_sents_for_selected_regions, reference_sents_for_selected_regions)

            generated_sents_for_selected_regions, generated_reports, removed_similar_generated_sentences = get_generated_reports(
                generated_sents_for_selected_regions,
                selected_regions,
                sentence_tokenizer,
                BERTSCORE_SIMILARITY_THRESHOLD
            )

            gen_and_ref_sentences["generated_sentences"].extend(generated_sents_for_selected_regions)
            gen_and_ref_sentences["generated_sentences_normal_selected_regions"].extend(gen_sents_for_normal_selected_regions)
            gen_and_ref_sentences["generated_sentences_abnormal_selected_regions"].extend(gen_sents_for_abnormal_selected_regions)
            gen_and_ref_sentences["reference_sentences"].extend(reference_sents_for_selected_regions)
            gen_and_ref_sentences["reference_sentences_normal_selected_regions"].extend(ref_sents_for_normal_selected_regions)
            gen_and_ref_sentences["reference_sentences_abnormal_selected_regions"].extend(ref_sents_for_abnormal_selected_regions)
            gen_and_ref_reports["generated_reports"].extend(generated_reports)
            gen_and_ref_reports["reference_reports"].extend(reference_reports)
            gen_and_ref_reports["removed_similar_generated_sentences"].extend(removed_similar_generated_sentences)

            update_gen_and_ref_sentences_for_regions(gen_and_ref_sentences, generated_sents_for_selected_regions, reference_sents_for_selected_regions, selected_regions)
            update_num_generated_sentences_per_image(gen_and_ref_sentences, selected_regions)

            if num_batch < NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE:
                update_gen_sentences_with_corresponding_regions(gen_sentences_with_corresponding_regions, generated_sents_for_selected_regions, selected_regions)

            if num_batch < num_batches_to_process_for_image_plotting:
                print("Plotting", num_batch)
                plot_detections_and_sentences_to_tensorboard(
                    writer,
                    num_batch,
                    10000,
                    images,
                    image_targets,
                    selected_regions,
                    detections,
                    class_detected,
                    reference_sentences,
                    generated_sents_for_selected_regions,
                )

    return (gen_sentences_with_corresponding_regions, gen_and_ref_reports)

if __name__ == "__main__":
    main()