import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directories for training data
train_dir = 'PlantVillage/train'

# Parameters
img_height, img_width = 256, 256
batch_size = 32

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)

# Load training images to get class indices
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Save class indices to a JSON file
class_indices = train_generator.class_indices
with open('class_indices.json', 'w') as json_file:
    json.dump(class_indices, json_file)

print(f"Class indices saved to 'class_indices.json': {class_indices}")
