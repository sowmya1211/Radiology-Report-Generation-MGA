"""
Specify the checkpoint_path, images_paths and generated_reports_txt_path in the main function
before running this script.

If you encounter any spacy-related errors, try upgrading spacy to version 3.5.3 and spacy-transformers to version 1.2.5
pip install -U spacy
pip install -U spacy-transformers
"""

from collections import defaultdict 

import albumentations as A
import cv2 
import numpy as np
import evaluate
import spacy
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from src.full_model.report_generation_model import ReportGenerationModel
from src.full_model.train_full_model import get_tokenizer
from src.dataset.constants import ANATOMICAL_REGIONS

# Set GPU Check 
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Set the CUDA device to use (e.g., GPU with index 0)
    device = torch.device("cuda")   # Now you can use CUDA operations
else:
    device = torch.device("cpu")
 
BERTSCORE_SIMILARITY_THRESHOLD = 0.9
IMAGE_INPUT_SIZE = 512
MAX_NUM_TOKENS_GENERATE = 300
NUM_BEAMS = 4
mean = 0.471  # see get_transforms in src/dataset/compute_mean_std_dataset.py
std = 0.302


def write_generated_reports_to_txt(images_paths, generated_sentences, generated_reports, generated_reports_txt_path):
    with open(generated_reports_txt_path, "w") as f:
        for image_path, report, gen_sentences in zip(images_paths, generated_reports,generated_sentences):
            f.write(f"Image path: {image_path}\n\n")
            f.write(f"Generated report: {report}\n\n")
            f.write(f"Generated Sentences for each prominent region:\n\n")
            for sent in gen_sentences:
                f.write(f'{sent["Region"]} : {sent["Sentence"]}\n')
            f.write("\n\n") 
            f.write("=" * 30)
            f.write("\n\n")


def remove_duplicate_generated_sentences(generated_report, bert_score, sentence_tokenizer):
    def check_gen_sent_in_sents_to_be_removed(gen_sent, similar_generated_sents_to_be_removed):
        for lists_of_gen_sents_to_be_removed in similar_generated_sents_to_be_removed.values():
            if gen_sent in lists_of_gen_sents_to_be_removed:
                return True

        return False 

    # since different (closely related) regions can have the same generated sentence, we first remove exact duplicates

    # use sentence tokenizer to separate the generated sentences
    gen_sents = sentence_tokenizer(generated_report).sents

    # convert spacy.tokens.span.Span object into str by using .text attribute
    gen_sents = [sent.text for sent in gen_sents]

    # remove exact duplicates using a dict as an ordered set
    # note that dicts are insertion ordered as of Python 3.7
    gen_sents = list(dict.fromkeys(gen_sents))

    # there can still be generated sentences that are not exact duplicates, but nonetheless very similar
    # e.g. "The cardiomediastinal silhouette is normal." and "The cardiomediastinal silhouette is unremarkable."
    # to remove these "soft" duplicates, we use bertscore

    # similar_generated_sents_to_be_removed maps from one sentence to a list of similar sentences that are to be removed
    similar_generated_sents_to_be_removed = defaultdict(list)

    for i in range(len(gen_sents)):
        gen_sent_1 = gen_sents[i]

        for j in range(i + 1, len(gen_sents)):
            if check_gen_sent_in_sents_to_be_removed(gen_sent_1, similar_generated_sents_to_be_removed):
                break

            gen_sent_2 = gen_sents[j]
            if check_gen_sent_in_sents_to_be_removed(gen_sent_2, similar_generated_sents_to_be_removed):
                continue

            bert_score_result = bert_score.compute(
                lang="en", predictions=[gen_sent_1], references=[gen_sent_2], model_type="distilbert-base-uncased"
            )

            if bert_score_result["f1"][0] > BERTSCORE_SIMILARITY_THRESHOLD:
                # remove the generated similar sentence that is shorter
                if len(gen_sent_1) > len(gen_sent_2):
                    similar_generated_sents_to_be_removed[gen_sent_1].append(gen_sent_2)
                else:
                    similar_generated_sents_to_be_removed[gen_sent_2].append(gen_sent_1)

    generated_report = " ".join(
        sent
        for sent in gen_sents
        if not check_gen_sent_in_sents_to_be_removed(sent, similar_generated_sents_to_be_removed)
    )

    return generated_report

'''NEW CODE START'''
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
    
    generated_sents_for_selected_regions = remove_duplicate_generated_sentences(generated_sents_for_selected_regions, bert_score, sentence_tokenizer)
    return generated_sents_for_selected_regions
'''NEW CODE END'''

def convert_generated_sentences_to_report(generated_sents_for_selected_regions, bert_score, sentence_tokenizer):
    generated_report = " ".join(sent for sent in generated_sents_for_selected_regions)

    generated_report = remove_duplicate_generated_sentences(generated_report, bert_score, sentence_tokenizer)
    return generated_report


def get_report_for_image(model, image_tensor, tokenizer, bert_score, sentence_tokenizer):
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

    '''NEW CODE START'''
    gen_sents = []
    for sent in generated_sents_for_selected_regions:
        sent = remove_duplicate_regionwise_sentences(sent,bert_score,sentence_tokenizer)
        gen_sents.append(sent)
    
    generated_sents_for_selected_regions = gen_sents
    '''NEW CODE END'''

    generated_report = convert_generated_sentences_to_report(
        generated_sents_for_selected_regions, bert_score, sentence_tokenizer
    )  

    def get_region_name(region_index: int):
        for i, region_name in enumerate(ANATOMICAL_REGIONS):
            if i == region_index:
                return region_name

    #Process sentences retrieved
    generated_sents_for_selected_regions = np.asarray(generated_sents_for_selected_regions)
    #Get sentences with region name
    selected_regions = selected_regions[0] 
    generated_senetences_with_regions = []
    index_gen_sentence = 0
    
    for region_index, region_selected_bool in enumerate(selected_regions):
        if region_selected_bool:
            region_name = get_region_name(region_index)
            gen_sent = generated_sents_for_selected_regions[index_gen_sentence]
            generated_senetences_with_regions.append({"Region": region_name, "Sentence": gen_sent})
            index_gen_sentence += 1

    return generated_senetences_with_regions, generated_report


def get_image_tensor(image_path):
    # cv2.imread by default loads an image with 3 channels
    # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # shape (3056, 2544)
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
    
    image_transformed_batch = image_transformed.unsqueeze(0)  # shape (1, 1, 512, 512)

    return image_transformed_batch


def get_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=torch.device(device=device),
    )

    # if there is a key error when loading checkpoint, try uncommenting down below
    # since depending on the torch version, the state dicts may be different
    
    model = ReportGenerationModel(pretrain_without_lm_model=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device, non_blocking=True)
    model.eval()

    del checkpoint

    return model


def main():
    #RUN 122 - Num_slots:2 
    checkpoint_path = "/home/miruna/ReportGeneration_SSS_24/rgrg+mdt/runs/full_model/run_122/checkpoints/checkpoint_val_loss_22.038_overall_steps_14010.pt"
    

    model = get_model(checkpoint_path) 

    print("Model instantiated.")

    # paths to the images that we want to generate reports for
    images_paths = [
        "/home/miruna/ReportGeneration_SSS_24/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10001401/s51065211/8061113f-c019f3ae-fd1b7c54-33e8690d-be838099.jpg", #Test image
        "/home/miruna/ReportGeneration_SSS_24/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11000183/s51967845/3b8571b4-1418c4eb-ddf2b4bc-5cb96d9b-3b99df84.jpg",
         "/home/miruna/ReportGeneration_SSS_24/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p11/p11000011/s51029426/ff213473-b64efa18-863f2bad-76181481-30bc30d7.jpg",
        "/home/miruna/ReportGeneration_SSS_24/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10008304/s50053244/eee6e206-f7bc49c7-563f869c-ee75184d-c81e2907.jpg",
        "/home/miruna/ReportGeneration_SSS_24/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10004235/s58604118/c24939ff-cf96a7e2-dcc4a608-a9f63b02-2b64eca1.jpg",
        "/home/miruna/ReportGeneration_SSS_24/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10002661/s53368584/2630b0b8-2d5af3a6-1a02c2ca-952e9535-b44c35ee.jpg" 
    ]

    generated_reports_txt_path = "/home/miruna/ReportGeneration_SSS_24/rgrg+mdt/Inferences - Sample Reports Generated/Inference-output.txt"
    generated_reports = []  
    generated_sentences = []

    bert_score = evaluate.load("bertscore") 
    sentence_tokenizer = spacy.load("en_core_web_trf")
    tokenizer = get_tokenizer()

    # if you encounter a spacy-related error, try upgrading spacy to version 3.5.3 and spacy-transformers to version 1.2.5
    # pip install -U spacy
    # pip install -U spacy-transformers

    for image_path in tqdm(images_paths):
        image_tensor = get_image_tensor(image_path)  # shape (1, 1, 512, 512)
        generated_sentence, generated_report = get_report_for_image(model, image_tensor, tokenizer, bert_score, sentence_tokenizer)
        generated_reports.append(generated_report)
        generated_sentences.append(generated_sentence)
    
    write_generated_reports_to_txt(images_paths, generated_sentences, generated_reports, generated_reports_txt_path)

 
if __name__ == "__main__":
    main()
