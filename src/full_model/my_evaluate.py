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
import tempfile
from typing import Dict, List

import evaluate
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
import spacy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from tqdm import tqdm

from src.CheXbert.src.constants import CONDITIONS
from src.CheXbert.src.label import label
from src.CheXbert.src.models.bert_labeler import bert_labeler
from src.dataset.constants import ANATOMICAL_REGIONS
from src.full_model.evaluate_full_model.cider.cider import Cider
from src.full_model.run_configurations import (
    BATCH_SIZE,
    NUM_BEAMS,
    MAX_NUM_TOKENS_GENERATE,
    NUM_BATCHES_OF_GENERATED_SENTENCES_TO_SAVE_TO_FILE,
    NUM_BATCHES_OF_GENERATED_REPORTS_TO_SAVE_TO_FILE,
    NUM_BATCHES_TO_PROCESS_FOR_LANGUAGE_MODEL_EVALUATION,
    NUM_IMAGES_TO_PLOT,
    BERTSCORE_SIMILARITY_THRESHOLD,
)

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


def get_generated_sentence_for_region(
    generated_sentences_for_selected_regions, selected_regions, num_img, region_index
) -> str:
    """
    Args:
        generated_sentences_for_selected_regions (List[str]): holds the generated sentences for all regions that were selected in the batch, i.e. of length "num_regions_selected_in_batch"
        selected_regions (Tensor[bool]): of shape [batch_size x 29], specifies for each region if it was selected to get a sentences generated (True) or not by the binary classifier for region selection.
        Ergo has exactly "num_regions_selected_in_batch" True values.
        num_img (int): specifies the image we are currently processing in the batch, its value is in the range [0, batch_size-1]
        region_index (int): specifies the region we are currently processing of a single image, its value is in the range [0, 28]

    Returns:
        str: generated sentence for region specified by num_img and region_index
    """
    selected_regions_flat = selected_regions.reshape(-1)
    cum_sum_true_values = np.cumsum(selected_regions_flat)

    cum_sum_true_values = cum_sum_true_values.reshape(selected_regions.shape)
    cum_sum_true_values -= 1

    index = cum_sum_true_values[num_img][region_index]

    return generated_sentences_for_selected_regions[index]


def transform_sentence_to_fit_under_image(sentence):
    """
    Adds line breaks and whitespaces such that long reference or generated sentence
    fits under the plotted image.
    Values like max_line_length and prefix_for_alignment were found by trial-and-error.
    """
    max_line_length = 60
    if len(sentence) < max_line_length:
        return sentence

    words = sentence.split()
    transformed_sent = ""
    current_line_length = 0
    prefix_for_alignment = "\n" + " " * 20
    for word in words:
        if len(word) + current_line_length > max_line_length:
            word = f"{prefix_for_alignment}{word}"
            current_line_length = -len(prefix_for_alignment)

        current_line_length += len(word)
        transformed_sent += word + " "

    return transformed_sent


def update_region_set_text(
    region_set_text,
    color,
    reference_sentences_img,
    generated_sentences_for_selected_regions,
    region_index,
    selected_regions,
    num_img,
):
    """
    Create a single string region_set_text like in the example below.
    Each update creates 1 paragraph for 1 region/bbox.
    The (b), (r) and (y) represent the colors of the bounding boxes (in this case blue, red and yellow).

    Example:

    (b):
      reference: Normal cardiomediastinal silhouette, hila, and pleura.
      generated: The mediastinal and hilar contours are unremarkable.

    (r):
      reference:
      generated: [REGION NOT SELECTED]

    (y):
      reference:
      generated: There is no pleural effusion or pneumothorax.

    (... continues for 3 more regions/bboxes, for a total of 6 per region_set)
    """
    region_set_text += f"({color}):\n"
    reference_sentence_region = reference_sentences_img[region_index]

    # in case sentence is too long
    reference_sentence_region = transform_sentence_to_fit_under_image(reference_sentence_region)

    region_set_text += f"  reference: {reference_sentence_region}\n"

    box_region_selected = selected_regions[num_img][region_index]
    if not box_region_selected:
        region_set_text += "  generated: [REGION NOT SELECTED]\n\n"
    else:
        generated_sentence_region = get_generated_sentence_for_region(
            generated_sentences_for_selected_regions, selected_regions, num_img, region_index
        )
        generated_sentence_region = transform_sentence_to_fit_under_image(generated_sentence_region)
        region_set_text += f"  generated: {generated_sentence_region}\n\n"

    return region_set_text


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
    num_batch,
    overall_steps_taken,
    images,
    image_targets,
    selected_regions,
    detections,
    class_detected,
    reference_sentences,
    generated_sentences_for_selected_regions,
):
    # pred_boxes_batch is of shape [batch_size x 29 x 4] and contains the predicted region boxes with the highest score (i.e. top-1)
    # they are sorted in the 2nd dimension, meaning the 1st of the 29 boxes corresponds to the 1st region/class,
    # the 2nd to the 2nd class and so on
    pred_boxes_batch = detections["top_region_boxes"]

    # image_targets is a list of dicts, with each dict containing the key "boxes" that contain the gt boxes of a single image
    # gt_boxes is of shape [batch_size x 29 x 4]
    gt_boxes_batch = torch.stack([t["boxes"] for t in image_targets], dim=0)

    # plot 6 regions at a time, as to not overload the image with boxes (except for region_set_5, which has 5 regions)
    # the region_sets were chosen as to minimize overlap between the contained regions (i.e. better visibility)
    region_set_1 = ["right lung", "right costophrenic angle", "left lung", "left costophrenic angle", "cardiac silhouette", "spine"]
    region_set_2 = ["right upper lung zone", "right mid lung zone", "right lower lung zone", "left upper lung zone", "left mid lung zone", "left lower lung zone"]
    region_set_3 = ["right hilar structures", "right apical zone", "left hilar structures", "left apical zone", "right hemidiaphragm", "left hemidiaphragm"]
    region_set_4 = ["trachea", "right clavicle", "left clavicle", "aortic arch", "abdomen", "right atrium"]
    region_set_5 = ["mediastinum", "svc", "cavoatrial junction", "carina", "upper mediastinum"]

    regions_sets = [region_set_1, region_set_2, region_set_3, region_set_4, region_set_5]

    # put channel dimension (1st dim) last (0-th dim is batch-dim)
    images = images.numpy().transpose(0, 2, 3, 1)

    for num_img, image in enumerate(images):

        gt_boxes_img = gt_boxes_batch[num_img]
        pred_boxes_img = pred_boxes_batch[num_img]
        class_detected_img = class_detected[num_img].tolist()
        reference_sentences_img = reference_sentences[num_img]

        for num_region_set, region_set in enumerate(regions_sets):
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()

            plt.imshow(image, cmap="gray")
            plt.axis("on")

            region_indices = [ANATOMICAL_REGIONS[region] for region in region_set]
            region_colors = ["b", "g", "r", "c", "m", "y"]

            if num_region_set == 4:
                region_colors.pop()

            region_set_text = ""

            for region_index, color in zip(region_indices, region_colors):
                # box_gt and box_pred are both [List[float]] of len 4
                box_gt = gt_boxes_img[region_index].tolist()
                box_pred = pred_boxes_img[region_index].tolist()
                box_region_detected = class_detected_img[region_index]

                plot_box(box_gt, ax, clr=color, linestyle="solid", region_detected=box_region_detected)

                # only plot predicted box if class was actually detected
                if box_region_detected:
                    plot_box(box_pred, ax, clr=color, linestyle="dashed")

                # region_set_text = update_region_set_text(
                #     region_set_text,
                #     color,
                #     reference_sentences_img,
                #     generated_sentences_for_selected_regions,
                #     region_index,
                #     selected_regions,
                #     num_img,
                # )

            title = get_plot_title(region_set, region_indices, region_colors, class_detected_img)
            ax.set_title(title)

            #plt.xlabel(region_set_text, loc="left")

            # using writer.add_figure does not correctly display the region_set_text in tensorboard
            # so instead, fig is first saved as a png file to memory via BytesIO
            # (this also saves the region_set_text correctly in the png when bbox_inches="tight" is set)
            # then the png is loaded from memory and the 4th channel (alpha channel) is discarded
            # finally, writer.add_image is used to display the image in tensorboard
            buf = io.BytesIO()
            fig.savefig(buf, bbox_inches="tight")
            buf.seek(0)
            im = Image.open(buf)
            im = np.asarray(im)[..., :3]

            writer_image_num = num_batch * BATCH_SIZE + num_img
            writer.add_image(
                f"img_{writer_image_num}_region_set_{num_region_set}",
                im,
                global_step=overall_steps_taken,
                dataformats="HWC",
            )
            
            im = Image.fromarray(im)
            im.save(f"images/img_{writer_image_num}_region_set_{num_region_set}.jpg")

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


def update_num_generated_sentences_per_image(
    gen_and_ref_sentences: dict,
    selected_regions: np.array
):
    """
    selected_regions is a boolean array of shape (batch_size x 29) that will have a True value for all regions that were selected and hence for which sentences were generated.
    Thus to get the number of generated sentences per image, we just have to add up the True value along axis 1 (i.e. along the region dimension)
    """
    num_gen_sents_per_image = selected_regions.sum(axis=1).tolist()  # indices is a list[int] of len(batch_size)
    gen_and_ref_sentences["num_generated_sentences_per_image"].extend(num_gen_sents_per_image)


def update_gen_and_ref_sentences_for_regions(
    gen_and_ref_sentences: dict,
    generated_sents_for_selected_regions: List[str],
    reference_sents_for_selected_regions: List[str],
    selected_regions: np.array
):
    """Updates the gen_and_ref_sentences dict for each of the 29 regions, i.e. appends the generated and reference sentences for the regions (if they exist)

    Args:
        gen_and_ref_sentences (dict):
        generated_sents_for_selected_regions (List[str]): has exactly num_regions_selected_in_batch generated sentences
        reference_sents_for_selected_regions (List[str]): has exactly num_regions_selected_in_batch reference sentences
        selected_regions (np.array([bool])): of shape batch_size x 29, has exactly num_regions_selected_in_batch True values
        that specify the regions for whom sentences were generated
    """
    index_gen_ref_sentence = 0

    # of shape (batch_size * 29)
    selected_regions_flat = selected_regions.reshape(-1)
    for curr_index, region_selected_bool in enumerate(selected_regions_flat):
        if region_selected_bool:
            region_index = curr_index % 29
            gen_sent = generated_sents_for_selected_regions[index_gen_ref_sentence]
            ref_sent = reference_sents_for_selected_regions[index_gen_ref_sentence]

            gen_and_ref_sentences[region_index]["generated_sentences"].append(gen_sent)
            gen_and_ref_sentences[region_index]["reference_sentences"].append(ref_sent)

            index_gen_ref_sentence += 1


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

        print("Generated Sentences Original: \n",gen_sents_single_image)
        
        gen_sents = []
        #Making few changes in generated sentences of a single image
        for sent in gen_sents_single_image:
            sent = remove_duplicate_regionwise_sentences(sent,bert_score,sentence_tokenizer)
            gen_sents.append(sent)
        
        gen_sents_single_image = gen_sents
        print("Generated Sentences Modified: \n",gen_sents_single_image)

        # concatenate generated sentences of a single image to a continuous string gen_report_single_image
        gen_report_single_image = " ".join(sent for sent in gen_sents_single_image)

        gen_report_single_image, similar_generated_sents_to_be_removed = remove_duplicate_generated_sentences(
            gen_report_single_image, bert_score, sentence_tokenizer
        )

        generated_reports.append(gen_report_single_image)
        removed_similar_generated_sentences.append(similar_generated_sents_to_be_removed)

    return gen_sents_single_image, generated_reports, removed_similar_generated_sentences


def get_ref_sentences_for_selected_regions(reference_sentences, selected_regions):
    """
    Args:
        reference_sentences (List[List[str]]): outer list has len batch_size, inner list has len 29 (the inner list holds all reference phrases of a single image)
        selected_regions ([batch_size x 29]): boolean array that has exactly "num_regions_selected_in_batch" True values
    """
    # array of shape [batch_size x 29]
    reference_sentences = np.asarray(reference_sentences)

    ref_sentences_for_selected_regions = reference_sentences[selected_regions]

    return ref_sentences_for_selected_regions.tolist()


def get_sents_for_normal_abnormal_selected_regions(region_is_abnormal, selected_regions, generated_sentences_for_selected_regions, reference_sentences_for_selected_regions):
    selected_region_is_abnormal = region_is_abnormal[selected_regions]
    # selected_region_is_abnormal is a bool array of shape [num_regions_selected_in_batch] that specifies if a selected region is abnormal (True) or normal (False)

    gen_sents_for_selected_regions = np.asarray(generated_sentences_for_selected_regions)
    ref_sents_for_selected_regions = np.asarray(reference_sentences_for_selected_regions)

    gen_sents_for_normal_selected_regions = gen_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
    gen_sents_for_abnormal_selected_regions = gen_sents_for_selected_regions[selected_region_is_abnormal].tolist()

    ref_sents_for_normal_selected_regions = ref_sents_for_selected_regions[~selected_region_is_abnormal].tolist()
    ref_sents_for_abnormal_selected_regions = ref_sents_for_selected_regions[selected_region_is_abnormal].tolist()

    return (
        gen_sents_for_normal_selected_regions,
        gen_sents_for_abnormal_selected_regions,
        ref_sents_for_normal_selected_regions,
        ref_sents_for_abnormal_selected_regions,
    )