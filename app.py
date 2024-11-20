import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask_cors import CORS
import numpy as np
import json

# Load your trained model
model = load_model('crop_disease_vgg16_model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse class indices to get labels
class_labels = {v: k for k, v in class_indices.items()}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the index route
@app.route('/')
def index():
    return render_template('index.html')

# Handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return json.dumps({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return json.dumps({'error': 'No selected file'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = image.load_img(filepath, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image to [0, 1]

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = class_labels[predicted_class[0]]

 # Fetch symptoms
        symptoms_data = {
           "Pepper__bell___Bacterial_spot": [
        "Dark, water-soaked spots on leaves.",
        "Yellowing of leaf margins.",
        "Small, dark, sunken lesions on fruits.",
        "Defoliation and reduced yield.",
        "Wilting and stunted growth."
    ],
    "Pepper__bell___healthy": [
        "No visible symptoms.",
        "Bright green leaves.",
        "Healthy fruit development.",
        "Sturdy stems.",
        "Normal growth patterns."
    ],
    "Potato___Early_blight": [
        "Dark brown spots on older leaves.",
        "Yellowing of leaf margins.",
        "Defoliation.",
        "Reduced tuber quality.",
        "Stunted plant growth."
    ],
    "Potato___Late_blight": [
        "Water-soaked spots on leaves.",
        "White mold on the undersides of leaves.",
        "Rapid wilting of plants.",
        "Dark, greasy spots on tubers.",
        "Rotting of tubers in storage."
    ],
    "Potato___healthy": [
        "No visible symptoms.",
        "Healthy, green leaves.",
        "Robust tuber development.",
        "Strong stems.",
        "Normal growth patterns."
    ],
    "Tomato_Bacterial_spot": [
        "Dark, water-soaked lesions on leaves.",
        "Yellowing of leaf edges.",
        "Small, dark spots on fruits.",
        "Leaf curling and wilting.",
        "Reduced fruit quality."
    ],
    "Tomato_Early_blight": [
        "Dark, concentric rings on leaves.",
        "Yellowing and dropping of lower leaves.",
        "Reduced fruit yield.",
        "Dark lesions on stems.",
        "Stunted plant growth."
    ],
    "Tomato_Late_blight": [
        "Large, irregularly shaped water-soaked spots.",
        "White mold on the underside of leaves.",
        "Brown lesions on stems.",
        "Rapid wilting of plants.",
        "Tubers rot in the ground."
    ],
    "Tomato_Leaf_Mold": [
        "Yellowing of leaves.",
        "Fuzzy greenish-gray mold on the underside.",
        "Leaf curling.",
        "Defoliation.",
        "Reduced fruit quality."
    ],
    "Tomato_Septoria_leaf_spot": [
        "Small, round spots with dark borders.",
        "Yellowing leaves.",
        "Defoliation.",
        "Reduced yield.",
        "Dark, sunken spots on stems."
    ],
    "Tomato_Spider_mites_Two_spotted_spider_mite": [
        "Fine webbing on leaves.",
        "Yellowing and stippling of leaves.",
        "Leaf drop.",
        "Stunted growth.",
        "Brown, crispy leaves."
    ],
    "Tomato__Target_Spot": [
        "Dark, concentric ring spots on leaves.",
        "Leaf drop.",
        "Reduced yield.",
        "Spots may appear on fruits.",
        "Stunted growth."
    ],
    "Tomato__Tomato_YellowLeaf__Curl_Virus": [
        "Yellowing of leaves.",
        "Curling and distortion of leaves.",
        "Stunted growth.",
        "Reduced fruit set.",
        "Plant wilting."
    ],
    "Tomato__Tomato_mosaic_virus": [
        "Mosaic patterns on leaves.",
        "Stunted growth.",
        "Leaf curling.",
        "Deformed fruits.",
        "Reduced yield."
    ],
    "Tomato_healthy": [
        "No visible symptoms.",
        "Healthy green leaves.",
        "Robust fruit development.",
        "Strong stems.",
        "Normal growth patterns."
    ]
           
        }
         # Get symptoms for the predicted label
        symptoms = symptoms_data.get(predicted_label, ["No symptoms available"])



        # Return the prediction and symptoms as JSON
        return json.dumps({
            'prediction': predicted_label,
            'symptoms': symptoms
        }), 200

    return json.dumps({'error': 'Failed to process image'}), 500
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
