import json
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('crop_disease_model.h5')

# Load class indices from the JSON file
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the class indices to get a dictionary of label names
class_labels = {v: k for k, v in class_indices.items()}

# Load and preprocess the input image
img_path = '/Users/shweta/Downloads/tomatos.jpg' 
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale like the training data

# Make prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)[0]

# Print the predicted disease label
print(f"The predicted disease is: {class_labels[predicted_class]}")
