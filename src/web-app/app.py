from flask import Flask, render_template, request
import os
import io
from PIL import Image
import base64
import pandas as pd
#from ml_model import predict  # Import your ML model function

from src.full_model.my_inference import infer


UPLOAD_FOLDER = '/home/miruna/ReportGeneration_SSS_24/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_jpeg_files_from_csv():
    # Read the CSV file
    #df = pd.read_csv('../rgrg-main/dataset-with-reference-reports-ForMDT/concatenated.csv')
    df = pd.read_csv('../rgrg+mdt/dataset-with-reference-reports/test.csv') 
    mimic_image_file_paths = df['mimic_image_file_path'].tolist()     
    #file = open("check.txt", "w")
    #file.write(str(mimic_image_file_paths))    
    return mimic_image_file_paths

def filter_row(target_value):
     #df = pd.read_csv('../rgrg-main/dataset-with-reference-reports-ForMDT/concatenated.csv')
    df = pd.read_csv('../rgrg+mdt/dataset-with-reference-reports/test.csv')
    filtered_rows = df[df['mimic_image_file_path'] == target_value]
    #print(filtered_rows.columns)
    return filtered_rows

def get_images():
    print(os.getcwd())
    image_directory = "./images"
    image_files = [file for file in os.listdir(image_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    images = []
    for x in image_files:
        #print(x)
        file_obj = open(image_directory+"/"+x, 'rb')
        image_data = file_obj.read()
        image_data_base64 = base64.b64encode(io.BytesIO(image_data).read()).decode()
        images.append(image_data_base64)
    return images

# Route for the home page
@app.route('/')
def index():
    # Get list of images in the uploads folder
    image_file = "/home/miruna/ReportGeneration_SSS_24/dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/files/p10/p10010150/s50055231/e7f21453-7956d79a-44e44614-fae8ff16-d174d1a0.jpg"
    return render_template('index.html', image_files=[image_file])

    # image_files = find_jpeg_files_from_csv()    
    # image_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if allowed_file(f)]
    # return render_template('index.html', image_files=image_files)

# Route to display the selected image and result
@app.route('/process', methods=['POST'])
def process_image():
    selected_image = request.form['selected_image']
    image_path = selected_image

    row = filter_row(image_path)
    gen_sentences_with_corresponding_regions, gen_and_ref_reports = infer(image_path)
    print("Generated and Reference Reports:\n",gen_and_ref_reports)
    print("Generated sentences and their corresponding regions:\n",gen_sentences_with_corresponding_regions)
    
    generated_reports = gen_and_ref_reports['generated_reports']
    removed_similar_generated_sentences = gen_and_ref_reports['removed_similar_generated_sentences']
    reference_reports = gen_and_ref_reports['reference_reports']
    gen_sentences_with_corresponding_regions = gen_sentences_with_corresponding_regions[0]
    
    region_sentence_pairs = [(region, sentence) for region, sentence in gen_sentences_with_corresponding_regions]

    
    # Pass the selected image to the ML model for processing
    result = "Output"#predict(image_path)  # Replace with your ML model function
    
    file_obj = open(image_path, 'rb')
    image_data = file_obj.read()
    image_data_base64 = base64.b64encode(io.BytesIO(image_data).read()).decode()
    
    images = get_images()
    
    parts = selected_image.split('/')
    subject_id = parts[-3][1:]  # Remove 'p' prefix
    study_id = parts[-2][1:]    # Remove 's' prefix
    image_id = parts[-1]
    
    return render_template('result.html', subject_id=subject_id,
                           study_id=study_id,
                           image_id=image_id,
                           selected_image=selected_image, 
                           result=result, 
                           image_data=image_data_base64, 
                           image_files = images, 
                           generated_reports=generated_reports, 
                           removed_similar_generated_sentences=removed_similar_generated_sentences,
                           reference_reports=reference_reports,
                           region_sentence_pairs=region_sentence_pairs)

    #return render_template('result.html', selected_image=selected_image, 
    #                       result=result, 
    #                       image_data=image_data_base64, 
    #                       image_files = images, 
    #                       generated_reports=generated_reports, 
    #                       removed_similar_generated_sentences=removed_similar_generated_sentences,
    #                       reference_reports=reference_reports,
    #                       region_sentence_pairs=region_sentence_pairs)

if __name__ == '__main__':
    app.run(debug=True)
