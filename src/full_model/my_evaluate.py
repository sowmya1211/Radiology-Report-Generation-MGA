"""
This module contains all functions used to evaluate the report generation model.

The (main) function evaluate_language_model of this module is called by the function evaluate_model in evaluate_model.py.

evaluate_language_model returns language_model_scores which include:
    - METEOR for:
        - all generated sentences
        - generated sentences for each region
        - generated sentences with gt = normal region (i.e. the region was considered normal by the radiologist)
        - generated sentences with gt = abnormal region (i.e. the region was considered abnormal by the radiologist)

    - BLEU 1-4, METEOR, ROUGE-L, CIDEr-D for all generated reports 
    - Clinical efficacy metrics for all generated reports:
        - micro-averaged over 5 observations
        - exampled-based averaged over all 14 observations
        - computed for each observation individually
"""
from collections import defaultdict
import csv
import io
import os
import re 
from typing import Dict, List

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from src.dataset.constants import ANATOMICAL_REGIONS

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

def get_plot_title(region_set, region_indices, region_colors, class_detected_img) -> str:
    """
    Get a plot title like in the below example.
    1 region_set always contains 6 regions (except for region_set_5, which has 5 regions).
    The characters in the brackets represent the colors of the corresponding bboxes (e.g. b = blue),
    "nd" stands for "not detected" in case the region was not detected by the object detector.

    right lung (b), right costophrenic angle (g, nd), left lung (r)
    left costophrenic angle (c), cardiac silhouette (m), spine (y, nd)
    """
    # get a list of 6 boolean values that specify if that region was detected
    class_detected = [class_detected_img[region_index] for region_index in region_indices]

    # add color_code to region name (e.g. "(r)" for red)
    # also add nd to the brackets if region was not detected (e.g. "(r, nd)" if red region was not detected)
    region_set = [
        region + f" ({color})" if cls_detect else region + f" ({color}, nd)"
        for region, color, cls_detect in zip(region_set, region_colors, class_detected)
    ]

    # add a line break to the title, as to not make it too long
    return ", ".join(region_set[:3]) + "\n" + ", ".join(region_set[3:])
def plot_box(box, ax, clr, linestyle, region_detected=True):
    x0, y0, x1, y1 = box
    h = y1 - y0
    w = x1 - x0
    ax.add_artist(
        plt.Rectangle(xy=(x0, y0), height=h, width=w, fill=False, color=clr, linewidth=1, linestyle=linestyle)
    )

    # add an annotation to the gt box, that the pred box does not exist (i.e. the corresponding region was not detected)
    if not region_detected:
        ax.annotate("not detected", (x0, y0), color=clr, weight="bold", fontsize=10)
def plot_detections_and_sentences_to_tensorboard(
    writer,
    overall_steps_taken,
    image_tensor,
    detections,
    class_detected,
):
    # pred_boxes_batch is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
    # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
    # the 2nd to the 2nd class and so on
    pred_boxes_batch = detections["top_region_boxes"]

    # plot 6 regions at a time, as to not overload the image with boxes (except for region_set_5, which has 5 regions)
    # the region_sets were chosen as to minimize overlap between the contained regions (i.e. better visibility)
    region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
    region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
    region_set_3 = ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"]
    region_set_4 = ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"]
    region_set_5 = ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]

    regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5]
    
    image_tensor = np.reshape(image_tensor, (IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE, 1))
    pred_boxes_img = pred_boxes_batch[0]
    class_detected_img = class_detected[0].tolist()

    for num_region_set, region_set in enumerate(regions_sets):
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()

        plt.imshow(image_tensor, cmap="gray")
        plt.axis("on")

        region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]
        region_colors = ["b", "g", "r", "c", "m", "y"]

        if num_region_set == 4:
            region_colors.pop()

        for region_index, color in zip(region_indices, region_colors):
            box_pred = pred_boxes_img[region_index].tolist()
            box_region_detected = class_detected_img[region_index]

            plot_box(box_pred, ax, clr=color, linestyle="solid",region_detected=box_region_detected)


        title = get_plot_title(region_set, region_indices, region_colors, class_detected_img)
        ax.set_title(title)

        buf = io.BytesIO()
        fig.savefig(buf, bbox_inches="tight")
        buf.seek(0)
        im = Image.open(buf)
        im = np.asarray(im)[..., :3]

        writer_image_num = 0
        writer.add_image(
            f"img_{writer_image_num}_region_set_{num_region_set}",
            im,
            global_step=overall_steps_taken,
            dataformats="HWC",
        )
        
        im = Image.fromarray(im)
        im.save(f"../Radiology-Report-Generation---MGA/src/web-app/images/img_{writer_image_num}_region_set_{num_region_set}.jpg")

        plt.close(fig)


def update_gen_sentences_with_corresponding_regions(
    gen_sentences_with_corresponding_regions,
    generated_sents_for_selected_regions,
    selected_regions
):
    """
    Args:
        gen_sentences_with_corresponding_regions (list[list[tuple[str, str]]]):
            len(outer_list)= (NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE * BATCH_SIZE),
            and inner list has len of how many regions were selected for a given image.
            Inner list hold tuples of (region_name, gen_sent), i.e. region name and its corresponding generated sentence
        generated_sentences_for_selected_regions (List[str]): of length "num_regions_selected_in_batch"
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
    """
    def get_region_name(region_index: int):
        for i, region_name in enumerate(ANATOMICAL_REGIONS):
            if i == region_index:
                return region_name

    index_gen_sentence = 0

    # selected_regions_single_image is a row with 29 bool values corresponding to a single image
    for selected_regions_single_image in selected_regions:
        gen_sents_with_regions_single_image = []

        for region_index, region_selected_bool in enumerate(selected_regions_single_image):
            if region_selected_bool:
                region_name = get_region_name(region_index)
                gen_sent = generated_sents_for_selected_regions[index_gen_sentence]

                gen_sents_with_regions_single_image.append((region_name, gen_sent))

                index_gen_sentence += 1

        gen_sentences_with_corresponding_regions.append(gen_sents_with_regions_single_image)


def get_generated_reports(generated_sentences_for_selected_regions, selected_regions, sentence_tokenizer, bertscore_threshold):
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): of length "num_regions_selected_in_batch"
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
        sentence_tokenizer: used in remove_duplicate_generated_sentences to separate the generated sentences

    Return:
        generated_reports (List[str]): list of length batch_size containing generated reports for every image in batch
        removed_similar_generated_sentences (List[Dict[str, List]): list of length batch_size containing dicts that map from one generated sentence to a list
        of other generated sentences that were removed because they were too similar. Useful for manually verifying if removing similar generated sentences was successful
    """

    def remove_duplicate_regionwise_sentences(generated_sents_for_selected_regions, bert_score, sentence_tokenizer):
        
        #Getting one regionwise sentence as input - Split it into one sentence each using '.'
        split_sentences = generated_sents_for_selected_regions.split('.')
        new_sent = []
        
        for sent in split_sentences:
            sent = sent.strip()
            if "comparison" or "Comparison" in sent: #In comparison to the previous radiograph, there has been a decrease in the size of pleural effusion
                if ',' in sent:
                    parts = sent.split(',',1)
                    if len(parts) > 1:
                        sent = parts[1].strip().capitalize() #There has been a decrease in the size of pleural effusion
                    if "has been" in sent:
                        sent = sent.replace("has been","is") 
            new_sent.append(sent)
        generated_sents_for_selected_regions = ". ".join(new_sent)
        generated_sents_for_selected_regions.replace(". ",".")
        
        generated_sents_for_selected_regions, _ = remove_duplicate_generated_sentences(generated_sents_for_selected_regions, bert_score, sentence_tokenizer)
        return generated_sents_for_selected_regions

    def remove_duplicate_generated_sentences(gen_report_single_image, bert_score,sentence_tokenizer):
        def check_gen_sent_in_sents_to_be_removed(gen_sent, similar_generated_sents_to_be_removed):
            for lists_of_gen_sents_to_be_removed in similar_generated_sents_to_be_removed.values():
                if gen_sent in lists_of_gen_sents_to_be_removed:
                    return True

            return False

        # since different (closely related) regions can have the same generated sentence, we first remove exact duplicates

        # use sentence tokenizer to separate the generated sentences
        gen_sents_single_image = sentence_tokenizer(gen_report_single_image).sents

        # convert spacy.tokens.span.Span object into str by using .text attribute
        gen_sents_single_image = [sent.text for sent in gen_sents_single_image]

        # remove exact duplicates using a dict as an ordered set
        # note that dicts are insertion ordered as of Python 3.7
        gen_sents_single_image = list(dict.fromkeys(gen_sents_single_image))

        # there can still be generated sentences that are not exact duplicates, but nonetheless very similar
        # e.g. "The cardiomediastinal silhouette is normal." and "The cardiomediastinal silhouette is unremarkable."
        # to remove these "soft" duplicates, we use bertscore

        # similar_generated_sents_to_be_removed maps from one sentence to a list of similar sentences that are to be removed
        similar_generated_sents_to_be_removed = defaultdict(list)

        # TODO:
        # the nested for loops below check each generated sentence with every other generated sentence
        # this is not particularly efficient, since e.g. generated sentences for the region "right lung" most likely
        # will never be similar to generated sentences for the region "abdomen"
        # thus, one could speed up these checks by only checking anatomical regions that are similar to each other

        for i in range(len(gen_sents_single_image)):
            gen_sent_1 = gen_sents_single_image[i]

            for j in range(i + 1, len(gen_sents_single_image)):
                if check_gen_sent_in_sents_to_be_removed(gen_sent_1, similar_generated_sents_to_be_removed):
                    break

                gen_sent_2 = gen_sents_single_image[j]
                if check_gen_sent_in_sents_to_be_removed(gen_sent_2, similar_generated_sents_to_be_removed):
                    continue

                bert_score_result = bert_score.compute(
                    lang="en", predictions=[gen_sent_1], references=[gen_sent_2], model_type="distilbert-base-uncased"
                )

                if bert_score_result["f1"][0] > bertscore_threshold:
                    # remove the generated similar sentence that is shorter
                    if len(gen_sent_1) > len(gen_sent_2):
                        similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                    else:
                        similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

        gen_report_single_image = " ".join(
            sent for sent in gen_sents_single_image if not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
        )

        return gen_report_single_image, similar_generated_sents_to_be_removed

    bert_score = evaluate.load("bertscore")
    
    generated_reports = []
    removed_similar_generated_sentences = []
    curr_index = 0

    for selected_regions_single_image in selected_regions:
        # sum up all True values for a single row in the array (corresponing to a single image)
        num_selected_regions_single_image = np.sum(selected_regions_single_image)

        # use curr_index and num_selected_regions_single_image to index all generated sentences corresponding to a single image
        gen_sents_single_image = generated_sentences_for_selected_regions[
            curr_index: curr_index + num_selected_regions_single_image
        ]

        # update curr_index for next image
        curr_index += num_selected_regions_single_image
        
        gen_sents = []
        #Making few changes in generated sentences of a single image
        for sent in gen_sents_single_image:
            sent = remove_duplicate_regionwise_sentences(sent,bert_score,sentence_tokenizer)
            gen_sents.append(sent)
        
        gen_sents_single_image = gen_sents

        # concatenate generated sentences of a single image to a continuous string gen_report_single_image
        gen_report_single_image = " ".join(sent for sent in gen_sents_single_image)

        gen_report_single_image, similar_generated_sents_to_be_removed = remove_duplicate_generated_sentences(
            gen_report_single_image, bert_score, sentence_tokenizer
        )

        generated_reports.append(gen_report_single_image)
        removed_similar_generated_sentences.append(similar_generated_sents_to_be_removed)

    return gen_sents_single_image, generated_reports, removed_similar_generated_sentences