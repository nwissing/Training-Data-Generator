import matplotlib
# prevent the application from running into a deadlock
matplotlib.use('Agg')


import cv2
import os
import argparse
import io
import numpy as np
import matplotlib.pyplot as plt
import torch
from flask import Flask, request, jsonify, send_file
import json
import base64

from flask_cors import CORS

app = Flask(__name__)
api_key = os.environ.get('API_KEY')
CORS(app)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input',
    default='image.jpg'
)
args = parser.parse_args()

if not os.path.exists('outputs'):
    os.makedirs('outputs')
    

def draw_mask(mask, ax, random_color=False):
    # Help function that draws the mask on the input image
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.7])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def segmentImageBox(image, input_box):
    # Load SAM
    sam_checkpoint = "sam_vit_l_0b3195.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # If the GPU is available, use it. If not, use the CPU
    model_type = "vit_l"
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Load image into the Image Encoder
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # Call the predictor.predict function with the provided input_box
    # Load the promts into Prompt Encoder and prodocues the mask with Mask Decoder
    masks, _, _ = predictor.predict(
        box=input_box,
        multimask_output=False
    )

    # Change the image size to 768x768:
    # Set the size of the image to 998x998 before to avoid the typical white border at matplotlib.
    new_image_size = (998, 998)
    plt.figure(figsize=(new_image_size[1] / 100, new_image_size[0] / 100), dpi=100)
    plt.imshow(image)
    draw_mask(masks[0], plt.gca())
    plt.axis('off')
    # Save the figure to a byte stream
    output_bytes = io.BytesIO()
    plt.savefig(output_bytes, format='png', bbox_inches='tight', pad_inches=0)
    output_bytes.seek(2)
    plt.close()

    np.save("OUTPUTimage.npy", image)
    np.save("OUTPUTmask.npy", masks)
    return output_bytes.getvalue()

def segmentImagePoint(image, input_points, input_labels):
    # Convert the Points into the right format
    data_array = [[item['x'], item['y']] for item in input_points]
    input_points = np.array(data_array)

    # Load SAM
    sam_checkpoint = "sam_vit_l_0b3195.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # If the GPU is available, use it. If not, use the CPU 
    model_type = "vit_l"
    from segment_anything import sam_model_registry, SamPredictor
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    
    # Load image into the Image Encoder
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    # Call the predictor.predict function with the provided input_points and input_labels
    # Load the promts into Prompt Encoder and prodocues the mask with Mask Decoder
    masks, _, _ = predictor.predict(
        point_coords=input_points,
        point_labels = input_labels,
        multimask_output=False
    )

    np.save("OUTPUTimage.npy", image)
    np.save("OUTPUTmask.npy", masks)

    # Change the image size to 768x768:
    # Set the size of the image to 998x998 before to avoid the typical white border at matplotlib. 
    new_image_size = (998, 998)
    plt.figure(figsize=(new_image_size[1] / 100, new_image_size[0] / 100), dpi=100)
    plt.imshow(image)
    draw_mask(masks[0], plt.gca())
    plt.axis('off')

    # Save the figure to a byte stream
    output_bytes = io.BytesIO()
    plt.savefig(output_bytes, format='png', bbox_inches='tight', pad_inches=0)
    output_bytes.seek(2)
    plt.close()

    return output_bytes.getvalue()


@app.route('/seg_image/box', methods=['POST'])
def seg_image():
    # Read the image file and convert it to an OpenCV image
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image file provided.'}), 400

    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    # Read the input_box values from the request
    input_box = request.form.get('input_box')
    try:
        input_box = json.loads(input_box)
        if len(input_box) != 4:
            raise ValueError('input_box should contain four values.')
        input_box = np.array(input_box)
    except Exception as e:
        return jsonify({'error': 'Invalid input_box format.'}), 400

    # Call segmentImageBox to process the image with the provided input_box
    processed_image_bytes = segmentImageBox(image, input_box)


    # Return the processed image as a byte stream with appropriate MIME type
    processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
    return jsonify({'image': processed_image_base64})

@app.route('/seg_image/point', methods=['POST'])
def seg_image_point():
    # Read the image file and convert it to an OpenCV image
    image_file = request.files.get('image')
    if not image_file:
        return jsonify({'error': 'No image file provided.'}), 400
    
    image_bytes = image_file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Read the input_point values from the request
    input_points = request.form.get('input_points')
    try:
        input_points = json.loads(input_points)
        # Check if input_points is a list of valid points
        if not isinstance(input_points, list) or any(len(point) != 2 for point in input_points):
            raise ValueError('input_points should be a list of points, where each point is represented as [x, y].')
        input_points = np.array(input_points)
    except Exception as e:
        return jsonify({'error': 'Invalid input_points format.'}), 400

    # Read the input_labels values from the request
    input_labels = request.form.get('input_labels')
    try:
        input_labels = json.loads(input_labels)
        input_labels = np.array(input_labels)
    except Exception as e:
        return jsonify({'error': 'Invalid input_labels format.'}), 400

    # Call segmentImage to process the image with the provided input_point and input_labels
    processed_image_bytes = segmentImagePoint(image, input_points, input_labels)

    # Return the processed image as a byte stream with appropriate MIME type
    processed_image_base64 = base64.b64encode(processed_image_bytes).decode('utf-8')
    return jsonify({'image': processed_image_base64})

@app.route('/download_npy', methods=['GET'])
def download_npy():
    # Read the OUTPUTimage.npy file from the directory
        with open('OUTPUTimage.npy', 'rb') as npy_file:
            # Return the .npy file as a byte stream with appropriate MIME type
            response = send_file(
                io.BytesIO(npy_file.read()),
                mimetype='application/octet-stream'
            )
            response.headers['Content-Disposition'] = 'attachment; filename=OUTPUTimage.npy'
            return response

@app.route('/download_mask_npy', methods=['GET'])
def download_mask_npy():
    # Read the OUTPUTmask.npy file from the directory
        with open('OUTPUTmask.npy', 'rb') as npy_file:
            # Return the .npy file as a byte stream with appropriate MIME type
            response = send_file(
                io.BytesIO(npy_file.read()),
                mimetype='application/octet-stream'
            )
            response.headers['Content-Disposition'] = 'attachment; filename=OUTPUTmask.npy'
            return response

@app.route('/get-api-key', methods=['GET'])
def get_api_key():
    return jsonify(api_key=api_key)

#@app.route('/')
#def hello_world():
#    return jsonify({'message': 'Hello, World!'})

@app.route('/')
def index():
    return send_file('index.html')

#if __name__ == '__main__':
#    app.run()

if __name__ == '__main__':
    app.run(host='0.0.0.0')
