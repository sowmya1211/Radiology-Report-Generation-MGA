from flask import Flask, render_template, request
import os
import io
from PIL import Image
import base64
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


from src.full_model.my_inference import infer


UPLOAD_FOLDER = '../Radiology-Report-Generation---MGA/dataset-with-reference-reports/images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
TEST_IMAGES_FOLDER = '../Radiology-Report-Generation---MGA/dataset-with-reference-reports'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_images():
    #print(os.getcwd())
    image_directory = "../Radiology-Report-Generation---MGA/src/web-app/images"
    image_files = [file for file in os.listdir(image_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    images = []
    for x in image_files:
        file_obj = open(image_directory+"/"+x, 'rb')
        image_data = file_obj.read()
        image_data_base64 = base64.b64encode(io.BytesIO(image_data).read()).decode()
        images.append(image_data_base64)
    return images

'''START Image Folder Dropdown '''
# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_jpeg_files_from_csv():
    # Read the CSV file
    df = pd.read_csv('../Radiology-Report-Generation---MGA/dataset-with-reference-reports/test-demo.csv') 
    df['mimic_image_file_path'] = df['mimic_image_file_path'].apply(lambda x: TEST_IMAGES_FOLDER + x)
    mimic_image_file_paths = df['mimic_image_file_path'].tolist()     
    return mimic_image_file_paths

def filter_row(target_value):
    df = pd.read_csv('../Radiology-Report-Generation---MGA/dataset-with-reference-reports/test-demo.csv')
    filtered_rows = df[df['mimic_image_file_path'] == target_value]
    return filtered_rows

# Route for the home page
@app.route('/')
def index():
    # Get list of images in the uploads folder
    image_files = find_jpeg_files_from_csv()    
    return render_template('index.html', image_files=image_files)
'''END Image Folder Dropdown '''

'''START Image Folder Upload '''
# # Route for the home page
# @app.route('/')
# def index():   
#     return render_template('index.html')
'''END Image Folder Upload '''

# Route to display the selected image and result
@app.route('/process', methods=['POST'])
def process_image():

    '''START Image Folder Dropdown '''
    selected_image = request.form['selected_image']
    image_path = selected_image
    row = filter_row(image_path)
    '''END Image Folder Dropdown '''

    '''START Image Folder Upload '''
    # if 'image' not in request.files:
    #     return "No file part"
    # file = request.files['image']
    # if file.filename == '':
    #     return "No selected image file"
    # if file:
    #     filename = file.filename
    #     image_directory = "../Radiology-Report-Generation---MGA/src/web-app/images"
    #     image_path = (image_directory + '/'+ filename)
    #     print(f"Image File saved at: {image_path}")
    '''END Image Folder Upload '''
  
    gen_sentences_with_corresponding_regions, gen_and_ref_reports = infer(image_path)
    print("Generated and Reference Reports:\n",gen_and_ref_reports)
    print("Generated sentences and their corresponding regions:\n",gen_sentences_with_corresponding_regions)
    
    generated_reports = gen_and_ref_reports['generated_reports']
    reference_reports = gen_and_ref_reports['reference_reports']
    gen_sentences_with_corresponding_regions = gen_sentences_with_corresponding_regions[0]
    
    region_sentence_pairs = [(region, sentence) for region, sentence in gen_sentences_with_corresponding_regions]

    
    # Pass the selected image to the ML model for processing
    result = "Output"
    
    file_obj = open(image_path, 'rb')
    image_data = file_obj.read()
    image_data_base64 = base64.b64encode(io.BytesIO(image_data).read()).decode()
    
    images = get_images()
    
    return render_template('result.html', 
                           image_path=image_path, 
                           result=result, 
                           image_data=image_data_base64, 
                           image_files = images, 
                           generated_reports=generated_reports, 
                           reference_reports=reference_reports,
                           region_sentence_pairs=region_sentence_pairs)

if __name__ == '__main__':
    app.run(debug=True)
