import os
import shutil
import random

# Set the paths to the main dataset and target directories
dataset_dir = 'PlantVillage'  # Path to your original dataset folder
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')

# Function to check if a file is hidden (for skipping .DS_Store and other system files)
def is_hidden(file):
    return file.startswith('.')  # Check if the file starts with a "."

# Create directories if not already present
for category in os.listdir(dataset_dir):
    category_path = os.path.join(dataset_dir, category)
    
    # Ignore non-folder files (hidden or other types) and already created directories
    if not os.path.isdir(category_path) or category in ['train', 'validation', 'test']:
        continue
    
    # Create corresponding subdirectories in train, validation, and test folders
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)
    
    # List and shuffle the images, filtering out hidden files
    images = [img for img in os.listdir(category_path) if not is_hidden(img)]
    random.shuffle(images)
    
    # Define split ratios
    train_split = int(0.7 * len(images))
    val_split = int(0.9 * len(images))
    
    # Split images into train, validation, and test sets
    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]
    
    # Move images to the respective directories
    for img in train_images:
        shutil.move(os.path.join(category_path, img), os.path.join(train_dir, category, img))
    for img in val_images:
        shutil.move(os.path.join(category_path, img), os.path.join(val_dir, category, img))
    for img in test_images:
        shutil.move(os.path.join(category_path, img), os.path.join(test_dir, category, img))

print("Dataset successfully split into train, validation, and test sets!")
